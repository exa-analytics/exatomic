import json

systems = [
    "ubuntu-latest",
    "macos-latest",
    "windows-latest"
]

versions = [
    "3.7",
    "3.8",
]

print(
    json.dumps(
        {
            "os": systems,
            "python-version": versions
        },
        separators=(",", ":")
    )
)
