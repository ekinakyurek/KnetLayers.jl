using Documenter, KLayers

makedocs(

    modules = [KLayers],
    clean = false,              # do we clean build dir
    format = :html,
    sitename = "KLayers.jl",
    authors = "Ekin Akyurek and contributors.",
    doctest = true,
    pages = Any[ # Compat: `Any` for 0.4 compat
        "Home" => "index.md",
        "Manual" => Any[
            "reference.md",
        ],
    ],
#    analytics = "UA-89508993-1",
#    linkcheck = !("skiplinks" in ARGS),
)

deploydocs(
    repo = "github.com/ekinakyurek/KLayers.jl.git",
    julia = "1.0",
    osname = "osx",
    target = "build",
    make = nothing,
    deps = nothing,
    #deps   = Deps.pip("mkdocs", "python-markdown-math"),
)
