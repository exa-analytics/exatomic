// Copright (c) 2015-2020, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
// Export widget models and views, and the npm package version number.
module.exports = {}

const base = require('./base')
const utils = require('./utils')
const appthree = require('./appthree')
const widgets = require('./widgets')
const tensor = require('./tensor')
const scene = require('./scene')
const util = require('./util')
const app = require('./app')

const loaded = [
    base, utils, appthree, widgets, tensor, scene, util, app,
]

Object.keys(loaded).forEach((key) => {
    const mod = loaded[key]
    Object.keys(mod).forEach((obj) => {
        module.exports[obj] = mod[obj]
    })
})

module.exports.version = require('../package.json').version
