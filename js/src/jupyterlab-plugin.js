// Copright (c) 2015-2020, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0

const base = require('@jupyter-widgets/base')
const exatomic = require('./index')

module.exports = {
    id: 'exatomic',
    requires: [base.IJupyterWidgetRegistry],
    activate(app, widgets) {
        widgets.registerWidget({
            name: 'exatomic',
            version: exatomic.version,
            exports: exatomic,
        })
    },
    autoStart: true,
}
