local script_path = debug.getinfo(1, "S").source:sub(2)
local script_dir = script_path:match("(.*)[/\\]")

local cleanup_fn = nil

local function run_spritegen()
    local success, result = pcall(dofile, script_dir .. "/plugin.lua")
    if not success then
        app.alert("Error loading SpriteGen: " .. tostring(result))
    elseif type(result) == "function" then
        cleanup_fn = result
    end
end

function init(plugin)
    plugin:newCommand{
        id = "SpriteGen",
        title = "SpriteGen",
        group = "file_scripts",
        onclick = run_spritegen
    }
end

function exit(plugin)
    if cleanup_fn then cleanup_fn() end
end
