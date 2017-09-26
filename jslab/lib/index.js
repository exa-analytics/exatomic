var jupyter_exatomic = require("jupyter-exatomic");

var jupyterlab_widgets = require("@jupyter-widgets/jupyterlab-manager");

module.exports = {
    id: "jupyter.extensions.jupyter-exatomic",
    requires: [jupyterlab_widgets.INBWidgetExtension],
    activate: function(app, widgets) {
        widgets.registerWidget({
            name: "jupyter-exatomic",
            version: jupyter_exatomic.version,
            exports: jupyter_exatomicc
        });
    },
    autoStart: true
};
