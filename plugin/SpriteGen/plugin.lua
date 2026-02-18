-- SpriteGen for Aseprite — Panel UI (WebSocket, non-blocking)
-- Uses Aseprite's native WebSocket + Timer for async communication

local script_path = debug.getinfo(1, "S").source:sub(2)
local script_dir = script_path:match("(.*)[/\\]")

local function safe_dofile(filename)
    local full_path = script_dir .. "/" .. filename
    local success, result = pcall(dofile, full_path)
    if not success then
        app.alert("Error loading " .. filename .. ": " .. tostring(result))
        return nil
    end
    return result
end

local json = safe_dofile("json.lua")
local base64 = safe_dofile("base64.lua")

if not json or not base64 then
    app.alert("Failed to load required libraries.")
    return
end

local WS_URL = "ws://127.0.0.1:5000/ws"

-- WebSocket and Timer state
local ws = nil
local ws_connected = false
local gen_timer = nil
local gen_start_time = 0
local panel_dlg = nil

-- State
local is_generating = false
local server_status = "Unknown"
local available_loras = {"None"}
local last_gen_time = 0
local last_seed = nil

local current_settings = {
    prompt = "pixel art, cute character, 16-bit style, vibrant colors",
    negative_prompt = "blurry, smooth, antialiased, realistic, photographic, 3d render, low quality",
    lora_model = "None",
    lora_strength = 0.8,
    pixel_width = 64,
    pixel_height = 64,
    colors = 16,
    steps = 25,
    guidance_scale = 7.5,
    seed = -1,
    use_current_sprite = false,
    img_strength = 0.6,
    remove_background = false,
    output_method = "New Layer"
}

-- Size presets
local size_presets = {
    {name = "32x32", width = 32, height = 32},
    {name = "64x64", width = 64, height = 64},
    {name = "128x128", width = 128, height = 128},
    {name = "256x256", width = 256, height = 256},
    {name = "64x96", width = 64, height = 96},
    {name = "96x64", width = 96, height = 64}
}

local function size_preset_names()
    local names = {}
    for _, p in ipairs(size_presets) do table.insert(names, p.name) end
    return names
end

local function format_time(s)
    if s < 60 then return string.format("%.1fs", s)
    else return string.format("%.1fm", s / 60) end
end

-- Place generated image into Aseprite
local function place_image(image_data, output_method)
    local image_mode = (image_data.mode == "rgba") and ColorMode.RGBA or ColorMode.RGB

    if not app.activeSprite then
        app.command.NewFile{
            width = image_data.width,
            height = image_data.height,
            colorMode = image_mode
        }
    end

    if not app.activeSprite.selection.isEmpty then
        app.command.Cancel()
    end

    local cel
    app.transaction("AI Generation", function()
        local layer = app.activeSprite:newLayer{
            name = "AI Gen " .. os.date("%H:%M:%S"),
            colorMode = image_mode
        }
        app.activeLayer = layer

        local frame
        if output_method == "New Frame" then
            frame = app.activeSprite:newEmptyFrame(app.activeFrame.frameNumber + 1)
        else
            frame = app.activeFrame
        end
        cel = app.activeSprite:newCel(layer, frame)
    end)

    if not cel then
        app.alert("Could not prepare canvas.")
        return
    end

    local pixel_data = base64.decode(image_data.base64)

    app.transaction("Place AI Image", function()
        local im = Image(image_data.width, image_data.height, image_mode)
        local ok = pcall(function() im.bytes = pixel_data end)
        if ok then
            cel.image:clear()
            cel.image:drawImage(im, Point(0, 0))
        else
            app.alert("Failed to process image data.")
        end
    end)

    app.refresh()
end

-- Stop generation timer and reset UI state
local function stop_generation()
    is_generating = false
    if gen_timer and gen_timer.isRunning then gen_timer:stop() end
    if panel_dlg then
        panel_dlg:modify{id="generate_btn", enabled=true, text="Generate Image"}
    end
end

-- WebSocket message handler (runs on UI thread, non-blocking)
local function on_ws_message(msg_type, data)
    if msg_type == WebSocketMessageType.OPEN then
        ws_connected = true
        -- Connection established, request status
        if ws then ws:sendText(json.encode({type = "status"})) end
        return
    end

    if msg_type == WebSocketMessageType.CLOSE then
        ws_connected = false
        server_status = "Disconnected"
        if panel_dlg then
            panel_dlg:modify{id="status_label", text="Status: Reconnecting..."}
        end
        return
    end

    if msg_type ~= WebSocketMessageType.TEXT then return end

    local ok, msg = pcall(json.decode, data)
    if not ok or not msg then return end

    if msg.type == "status" then
        server_status = "Online"
        if msg.current_model then
            server_status = server_status .. " - " .. msg.current_model
        end
        available_loras = msg.available_loras or {"None"}
        if panel_dlg then
            panel_dlg:modify{id="status_label", text="Status: " .. server_status}
            panel_dlg:modify{id="lora_model", options=available_loras, option=current_settings.lora_model}
        end

    elseif msg.type == "generating" then
        -- Server acknowledged the request, timer is already running

    elseif msg.type == "result" then
        last_gen_time = os.clock() - gen_start_time
        last_seed = msg.seed
        stop_generation()

        if msg.success and msg.image then
            place_image(msg.image, current_settings.output_method)
        end

        if panel_dlg then
            local info = "Last: " .. format_time(last_gen_time)
            if last_seed then info = info .. " | Seed: " .. tostring(last_seed) end
            panel_dlg:modify{id="info_label", text=info}
        end

    elseif msg.type == "error" then
        stop_generation()
        app.alert("Generation error:\n" .. tostring(msg.error))
    end
end

-- Request server status via WebSocket
local function fetch_status()
    if ws and ws_connected then
        ws:sendText(json.encode({type = "status"}))
    elseif panel_dlg then
        panel_dlg:modify{id="status_label", text="Status: Not connected"}
    end
end

-- Generate image (non-blocking via WebSocket)
local function generate_image(dlg)
    if is_generating then
        app.alert("Generation already in progress.")
        return
    end

    if not ws or not ws_connected then
        app.alert("Not connected to server.")
        return
    end

    if not current_settings.prompt or current_settings.prompt == "" then
        app.alert("Please enter a prompt.")
        return
    end

    is_generating = true
    gen_start_time = os.clock()
    dlg:modify{id="generate_btn", enabled=false, text="Generating... 0s"}

    -- Timer that updates the button every second
    if not gen_timer then
        gen_timer = Timer{interval=1.0, ontick=function()
            if is_generating and panel_dlg then
                local elapsed = os.clock() - gen_start_time
                panel_dlg:modify{id="generate_btn", text="Generating... " .. format_time(elapsed)}
            end
        end}
    end
    gen_timer:start()

    -- Build request data
    local request_data = {
        type = "generate",
        prompt = current_settings.prompt,
        negative_prompt = current_settings.negative_prompt,
        pixel_width = current_settings.pixel_width,
        pixel_height = current_settings.pixel_height,
        steps = current_settings.steps,
        guidance_scale = current_settings.guidance_scale,
        colors = current_settings.colors,
        lora_model = current_settings.lora_model,
        lora_strength = current_settings.lora_strength,
        remove_background = current_settings.remove_background
    }

    if current_settings.seed ~= -1 then
        request_data.seed = current_settings.seed
    end

    -- img2img: send current sprite as init_image, output size = sprite size
    if current_settings.use_current_sprite and app.activeSprite then
        local sprite = app.activeSprite
        local flat = Image(sprite.width, sprite.height, ColorMode.RGBA)
        flat:drawSprite(sprite, app.activeFrame.frameNumber)
        local encoded = base64.encode(flat.bytes)
        request_data.init_image = {
            base64 = encoded,
            width = sprite.width,
            height = sprite.height,
            mode = "rgba"
        }
        request_data.strength = current_settings.img_strength
        request_data.pixel_width = sprite.width
        request_data.pixel_height = sprite.height
    end

    -- Send via WebSocket (returns immediately, non-blocking)
    ws:sendText(json.encode(request_data))
end

-- Build the panel dialog
local function create_panel()
    local dlg = Dialog{title="SpriteGen", notitlebar=false}

    -- Status
    dlg:label{id="status_label", text="Status: " .. server_status}
    dlg:button{text="Refresh", onclick=function() fetch_status() end}
    dlg:separator{}

    -- Prompt
    dlg:entry{
        id="prompt",
        label="Prompt:",
        text=current_settings.prompt,
        onchange=function() current_settings.prompt = dlg.data.prompt end
    }
    dlg:entry{
        id="negative_prompt",
        label="Negative:",
        text=current_settings.negative_prompt,
        onchange=function() current_settings.negative_prompt = dlg.data.negative_prompt end
    }
    dlg:separator{}

    -- LoRA (populated from server status)
    dlg:combobox{
        id="lora_model",
        label="LoRA:",
        options=available_loras,
        option=current_settings.lora_model,
        onchange=function() current_settings.lora_model = dlg.data.lora_model end
    }

    dlg:slider{
        id="lora_strength",
        label="Strength:",
        min=0,
        max=200,
        value=math.floor(current_settings.lora_strength * 100),
        onchange=function() current_settings.lora_strength = dlg.data.lora_strength / 100 end
    }

    dlg:separator{}

    -- Pixel art output size
    dlg:combobox{
        id="size_preset",
        label="Size:",
        options=size_preset_names(),
        option=current_settings.pixel_width .. "x" .. current_settings.pixel_height,
        onchange=function()
            for _, p in ipairs(size_presets) do
                if p.name == dlg.data.size_preset then
                    current_settings.pixel_width = p.width
                    current_settings.pixel_height = p.height
                    break
                end
            end
        end
    }

    dlg:number{
        id="colors",
        label="Colors:",
        text=tostring(current_settings.colors),
        decimals=0,
        onchange=function() current_settings.colors = dlg.data.colors end
    }

    dlg:slider{
        id="steps",
        label="Steps:",
        min=1,
        max=50,
        value=current_settings.steps,
        onchange=function() current_settings.steps = dlg.data.steps end
    }

    dlg:slider{
        id="guidance_scale",
        label="Guidance:",
        min=1,
        max=20,
        value=math.floor(current_settings.guidance_scale),
        onchange=function() current_settings.guidance_scale = dlg.data.guidance_scale end
    }

    dlg:number{
        id="seed",
        label="Seed:",
        text=tostring(current_settings.seed),
        decimals=0,
        onchange=function() current_settings.seed = dlg.data.seed end
    }

    dlg:separator{}

    -- img2img: use current sprite as input
    dlg:check{
        id="use_current_sprite",
        text="Use Current Sprite (img2img)",
        selected=current_settings.use_current_sprite,
        onclick=function() current_settings.use_current_sprite = dlg.data.use_current_sprite end
    }

    dlg:slider{
        id="img_strength",
        label="Change:",
        min=10,
        max=100,
        value=math.floor(current_settings.img_strength * 100),
        onchange=function() current_settings.img_strength = dlg.data.img_strength / 100 end
    }

    -- Options
    dlg:check{
        id="remove_background",
        text="Remove Background",
        selected=current_settings.remove_background,
        onclick=function() current_settings.remove_background = dlg.data.remove_background end
    }

    dlg:combobox{
        id="output_method",
        label="Output:",
        options={"New Layer", "New Frame"},
        option=current_settings.output_method,
        onchange=function() current_settings.output_method = dlg.data.output_method end
    }

    dlg:separator{}

    -- Generate
    dlg:button{
        id="generate_btn",
        text="Generate Image",
        focus=true,
        onclick=function() generate_image(dlg) end
    }

    dlg:label{id="info_label", text="Ready to generate"}

    dlg:show{wait=false}

    -- Store dialog reference for callbacks
    panel_dlg = dlg

    -- Open WebSocket with auto-reconnect
    ws = WebSocket{
        url = WS_URL,
        onreceive = on_ws_message,
        deflate = false,
        minreconnectwait = 1,
        maxreconnectwait = 10
    }
    ws:connect()
end

create_panel()

-- Return cleanup function for main.lua exit()
return function()
    if gen_timer and gen_timer.isRunning then gen_timer:stop() end
    if ws then ws:close() end
    ws_connected = false
end
