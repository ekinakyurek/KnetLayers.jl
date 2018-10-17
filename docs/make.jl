using Documenter, KnetLayers

makedocs(

    modules = [KnetLayers],
    clean = false,              # do we clean build dir
    format = :html,
    sitename = "KnetLayers.jl",
    authors = "Ekin Akyurek and contributors.",
    doctest = true,
    pages = Any[ # Compat: `Any` for 0.4 compat
        "Home" => "index.md",
        "Function Documentation" => Any[
            "reference.md",
        ],
    ],
#    analytics = "UA-89508993-1",
#    linkcheck = !("skiplinks" in ARGS),
)

deploydocs(
    repo = "github.com/ekinakyurek/KnetLayers.jl.git",
    julia = "1.0",
    osname = "linux",
    target = "build",
    make = nothing,
    deps = nothing,
    #deps   = Deps.pip("mkdocs", "python-markdown-math"),
)
