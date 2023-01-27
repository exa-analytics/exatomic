"""
Dynamic support matrix builder for CICD
"""
import json

systems = [
    "ubuntu-latest",
    #"macos-latest",
    #"windows-latest"
]

versions = [
    #"3.7",
    "3.8",
    #"3.9",
    #"3.10",
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
