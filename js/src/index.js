// Copright (c) 2015-2020, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
// Export widget models and views, and the npm package version number.
module.exports = {}

const loaded = [
    require('./base'),
    require('./utils'),
    require('./appthree'),
    require('./widgets'),
    require('./tensor'),
]

Object.keys(loaded).forEach((key) => {
    const mod = loaded[key]
    Object.keys(mod).forEach((obj) => {
        module.exports[obj] = mod[obj]
    })
})

module.exports.version = require('../package.json').version
