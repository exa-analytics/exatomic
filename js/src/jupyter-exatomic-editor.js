"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __assign = (this && this.__assign) || Object.assign || function(t) {
    for (var s, i = 1, n = arguments.length; i < n; i++) {
        s = arguments[i];
        for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
            t[p] = s[p];
    }
    return t;
};
exports.__esModule = true;
var base = require("./jupyter-exatomic-base.js");
var three = require("three");
var EditorSceneModel = /** @class */ (function (_super) {
    __extends(EditorSceneModel, _super);
    function EditorSceneModel() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    EditorSceneModel.prototype.defaults = function () {
        return __assign({}, _super.prototype.defaults.call(this), {
            '_view_name': "EditorSceneView",
            '_model_name': "EditorSceneModel"
        });
    };
    return EditorSceneModel;
}(base.ExatomicSceneModel));
exports.EditorSceneModel = EditorSceneModel;
var EditorSceneView = /** @class */ (function (_super) {
    __extends(EditorSceneView, _super);
    function EditorSceneView() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    EditorSceneView.prototype.init = function () {
        _super.prototype.init.call(this);
        // Is it a problem, reassigning this?
        this.three_promises = this.three_promises.then(this.worker.bind(this));
    };
    EditorSceneView.prototype.worker = function () {
        console.log("in worker....");
        console.log(this);
        console.log(this.app3d);
        console.log(this.app3d.scene);
        var that = this;
        this.el.addEventListener('click', that.clicked);
        console.log("added listener");
    };
    EditorSceneView.prototype.add_sphere = function (x, y) {
        var geometry = new three.SphereGeometry(2, 32, 32);
        var material = new three.MeshPhongMaterial(color = "blue");
        var sphere = new three.Mesh(geometry, material);
        sphere.position = new three.Vector3(x, y, 0);
        console.log(sphere);
        this.app3d.meshes['editor'] = [sphere];
        this.app3d.add_meshes('editor');
    };
    EditorSceneView.prototype.del_sphere = function (x, y) {
        console.log("Good luck deleting....");
    };
    EditorSceneView.prototype.clicked = function (e) {
        console.log("clicked!");
        switch (e.button) {
            case 0:
                this.add_sphere(e.clientX, e.clientY);
            case 2:
                this.del_sphere(e.clientX, e.clientY);
        }
    };
    return EditorSceneView;
}(base.ExatomicSceneView));
exports.EditorSceneView = EditorSceneView;
