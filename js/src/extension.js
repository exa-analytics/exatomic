// Copright (c) 2015-2020, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
// This file contains the javascript that is run when the notebook is loaded.
// It contains some requirejs configuration and the `load_ipython_extension`
// which is required for any notebook extension.

// Configure requirejs
if (window.require) {
    window.require.config({
        map: {
            '*': {
                exatomic: 'nbextensions/exatomic/index',
            },
        },
    })
}

// Export the required load_ipython_extention
module.exports = {
    load_ipython_extension() {},
}
