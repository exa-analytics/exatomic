// Copright (c) 2015-2020, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/* """
=================
scene.ts
=================
A 3D scene for exatomic

*/

import * as widgets from '@jupyter-widgets/base'
import * as pkg from '../package.json'

// const version: string = require('../package.json').version
const semver: string = `^${pkg.version}`

export class SceneModel extends widgets.DOMWidgetModel {
    defaults() {
        return {
            ...super.defaults(),
            _model_name: 'SceneModel',
            _view_name: 'SceneView',
            _model_module_version: semver,
            _view_module_version: semver,
            _model_module: 'exatomic',
            _view_module: 'exatomic',
        }
    }
}


export class SceneView extends widgets.DOMWidgetView {
    initialize(parameters: any) {
        console.log("SceneView initialize")
        console.log(typeof parameters)
        super.initialize(parameters)
        this.initListeners()
    }
    initListeners() {
        this.listenTo(this.model, 'change:flag', this.updateFlag)
    }
    updateFlag() {
        console.log('flag was updated')
    }
}
