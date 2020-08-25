// Copright (c) 2015-2020, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/* """
=================
widgets.js
=================
JavaScript 'frontend' complement of exatomic's Container for use within
the Jupyter notebook interface.
*/

const base = require('./base')
const utils = require('./utils')
const three = require('./appthree')

export class UniverseSceneModel extends base.ExatomicSceneModel {
    defaults() {
        return {
            ...super.defaults(),
            _model_name: 'UniverseSceneModel',
            _view_name: 'UniverseSceneView',
        }
    }
}

export class UniverseSceneView extends base.ExatomicSceneView {
    init() {
        window.addEventListener('resize', this.resize.bind(this))
        this.app3d = new three.App3D(this)
        this.three_promises = this.app3d.init_promise()
        this.promises = Promise.all([
            utils.fparse(this, 'atom_x'),
            utils.fparse(this, 'atom_y'),
            utils.fparse(this, 'atom_z'),
            utils.fparse(this, 'atom_s'),
            utils.mesolv(this, 'atom_cr'),
            utils.mesolv(this, 'atom_vr'),
            utils.mesolv(this, 'atom_c'),
            utils.mesolv(this, 'atom_l'),
            utils.fparse(this, 'two_b0'),
            utils.fparse(this, 'two_b1'),
            utils.mesolv(this, 'field_i'),
            utils.mesolv(this, 'field_p'),
            utils.mesolv(this, 'field_v'),
            utils.mesolv(this, 'tensor_d'),
            utils.mesolv(this, 'freq_d'),
        ])
        this.three_promises = this.app3d.finalize(this.three_promises)
            .then(this.addAtom.bind(this))
            .then(this.app3d.set_camera_from_scene.bind(this.app3d))
    }

    render() {
        return Promise.all([this.three_promises, this.promises])
    }

    addAtom() {
        this.app3d.clear_meshes('atom')
        this.app3d.clear_meshes('two')
        const fdx = this.model.get('frame_idx')
        const syms = this.atom_s[fdx]
        const colrs = utils.mapper(syms, this.atom_c)
        let atom; let bond; let radii; const
            r = ((this.model.get('bond_r') > 0) ? this.model.get('bond_r') : 0.15)
        if (this.model.get('atom_3d')) {
            switch (this.model.get('fill_idx')) {
            case 0:
                radii = utils.mapper(syms, this.atom_cr).map((x) => x * 0.5)
                atom = this.app3d.add_spheres
                bond = this.app3d.add_cylinders
                break
            case 1:
                radii = utils.mapper(syms, this.atom_vr)
                atom = this.app3d.add_spheres
                bond = null
                break
            case 2:
                radii = utils.mapper(syms, this.atom_cr)
                atom = this.app3d.add_spheres
                bond = this.app3d.add_cylinders
                break
                // case 3:
                //    radii = utils.mapper(syms, this.atom_cr)
                //                    .map(function(x) { return x * 0.5; });
                //    atom = this.app3d.add_points;
                //    bond = this.app3d.add_lines;
                //    break;
                // case 4:
                //    r = 0.6
                //    radii = r + 0.05;
                //    atom = this.app3d.add_spheres;
                //    bond = this.app3d.add_cylinders;
                //    break;
                // TODO : default case
            default:
                radii = utils.mapper(syms, this.atom_cr)
                atom = this.app3d.add_points
                bond = null
                break
            }
        } else {
            if (this.app3d.selected.length > 0) {
                this.clearSelected()
            }
            radii = utils.mapper(syms, this.atom_cr)
                .map((x) => x * 0.5)
            atom = this.app3d.add_points
            bond = this.app3d.add_lines
        }
        let labels = utils.mapper(syms, this.atom_l)
        const a = []
        Object.entries(labels).forEach((e) => {
            a.push(e[1] + e[0].toString())
        })
        labels = a
        this.app3d.meshes.atom = atom(
            this.atom_x[fdx], this.atom_y[fdx],
            this.atom_z[fdx], colrs, radii, labels,
        )
        if (this.two_b0.length !== 0) {
            this.app3d.meshes.two = (bond != null) ? bond(
                this.two_b0[fdx], this.two_b1[fdx],
                this.atom_x[fdx], this.atom_y[fdx],
                this.atom_z[fdx], colrs, r,
            ) : null
        }
        this.app3d.add_meshes()
    }

    addField() {
        this.app3d.clear_meshes('field')
        const fldx = this.model.get('field_idx')
        const fdx = this.model.get('frame_idx')
        const fps = this.field_p[fdx][fldx]
        if ((!this.model.get('field_show'))
            || (fldx === 'null')
            || (typeof fps === 'undefined')) { return }
        const idx = this.field_i[fdx][fldx]
        const that = this
        if (typeof this.field_v[idx] === 'string') {
            utils.jsonparse(this.field_v[idx])
                .then((values) => {
                    that.field_v[idx] = values
                    that.app3d.meshes.field = that.app3d.add_scalar_field(
                        utils.scalar_field(
                            utils.gen_field_arrays(fps),
                            values,
                        ),
                        that.model.get('field_iso'),
                        that.model.get('field_o'), 2,
                        that.colors(),
                    )
                    that.app3d.add_meshes('field')
                })
        } else {
            this.app3d.meshes.field = this.app3d.add_scalar_field(
                utils.scalar_field(
                    utils.gen_field_arrays(fps),
                    this.field_v[idx],
                ),
                this.model.get('field_iso'),
                this.model.get('field_o'), 2,
                this.colors(),
            )
            this.app3d.add_meshes('field')
        }
    }

    addContour() {
        this.app3d.clear_meshes('contour')
        const fldx = this.model.get('field_idx')
        // Specifically test for string null
        const fdx = this.model.get('frame_idx')
        const idx = this.field_i[fdx][fldx]
        const fps = this.field_p[fdx][fldx]
        if ((!this.model.get('cont_show'))
            || (fldx === 'null')
            || (typeof fps === 'undefined')) { return }
        const that = this
        if (typeof this.field_v[idx] === 'string') {
            utils.jsonparse(this.field_v[idx])
                .then((values) => {
                    that.field_v[idx] = values
                    that.app3d.meshes.contour = that.app3d.add_contour(
                        utils.scalar_field(
                            utils.gen_field_arrays(fps),
                            values,
                        ),
                        that.model.get('cont_num'),
                        that.model.get('cont_lim'),
                        that.model.get('cont_axis'),
                        that.model.get('cont_val'),
                        that.colors(),
                    )
                    that.app3d.add_meshes('contour')
                })
        } else {
            this.app3d.meshes.contour = this.app3d.add_contour(
                utils.scalar_field(
                    utils.gen_field_arrays(fps),
                    this.field_v[idx],
                ),
                this.model.get('cont_num'),
                this.model.get('cont_lim'),
                this.model.get('cont_axis'),
                this.model.get('cont_val'),
                this.colors(),
            )
            that.app3d.add_meshes('contour')
        }
    }

    addAxis() {
        // Additionally adds the unit cell
        this.app3d.clear_meshes('generic')
        if (this.model.get('axis')) {
            this.app3d.meshes.generic = this.app3d.add_unit_axis(
                this.model.get('atom_3d'),
            )
        }
        this.app3d.add_meshes('generic')
    }

    getTensor(fdx, tdx) {
        const t = this.tensor_d[fdx][tdx]
        return [
            [t.xx, t.xy, t.xz],
            [t.yx, t.yy, t.yz],
            [t.zx, t.zy, t.zz],
        ]
    }

    colorTensor() {
        const tdx = this.model.get('tidx')
        const fdx = this.model.get('frame_idx')
        let color
        Object.keys(this.tensor_d[fdx]).forEach((tensor) => {
            if (parseInt(tensor, 10) === tdx) {
                color = 0xafafaf
            } else {
                color = 0x000000
            }
            if (this.model.get('tens')) {
                this.app3d.meshes[`tensor${tensor}`][0].children[0].material.color.setHex(color)
            }
        })
    }

    addTensor() {
        const scaling = this.model.get('scale')
        const fdx = this.model.get('frame_idx')
        Object.keys(this.tensor_d[fdx]).forEach((tdx) => {
            this.app3d.clear_meshes(`tensor${tdx}`)
            const adx = this.tensor_d[fdx][tdx].atom
            if (this.model.get('tens')) {
                this.app3d.meshes[`tensor${tdx}`] = this.app3d.add_tensor_surface(
                    this.getTensor(fdx, tdx),
                    this.colors(),
                    this.atom_x[fdx][adx],
                    this.atom_y[fdx][adx],
                    this.atom_z[fdx][adx],
                    scaling,
                    `${this.tensor_d[fdx][tdx].label} tensor ${tdx.toString()}`,
                )
            }
            this.app3d.add_meshes(`tensor${tdx}`)
        })
    }

    visualize_freq: function() {
        var freqdx = this.model.get("freq_idx");
        var fdx = this.model.get("frame_idx");
        var scale = this.model.get("freq_scale");
        console.log(this.freq_d[freqdx]);
        for ( var property in this.freq_d[freqdx] ) {
            var dx = this.freq_d[freqdx][property]["dx"]
            var dy = this.freq_d[freqdx][property]["dy"]
            var dz = this.freq_d[freqdx][property]["dz"]
            var adx = this.freq_d[freqdx][property]["label"]
            this.app3d.clear_meshes("normmode"+adx);
            var atom_x = this.atom_x[fdx][adx]
            var atom_y = this.atom_y[fdx][adx]
            var atom_z = this.atom_z[fdx][adx]
            this.app3d.meshes["normmode"+adx] = this.app3d.add_freq_disp(freqdx, dx, dy,
                                                                         dz, atom_x, atom_y,
                                                                         atom_z, scale);
            this.app3d.add_meshes("normmode"+adx);
            //console.log(dx, dy, dz, property, atom_x, atom_y, atom_z, adx);
        }
        //console.log(this.app3d.meshes("normmode0"));
        console.log("Inside frequency shit");
        console.log(freqdx);
    },
    // events: {
    //     'click': 'handleClick'
    // },

    //handleClick(event) {
    //    // Handles click event to write data to a Python traitlet value
    //    event.preventDefault()
    //    const idx = this.app3d.selected.map((obj) => obj.name)
    //    const type = this.app3d.selected.map((obj) => obj.geometry.type)
    //    this.model.set('selected', { idx, type }) // Types must match exactly
    //    this.touch()
    //}

    clearSelected() {
        this.app3d.reset_colors()
        this.app3d.selected = []
        this.model.set('selected', {})
        this.touch()
    }

    initListeners() {
        super.initListeners()
        this.listenTo(this.model, 'change:frame_idx', this.addAtom)
        this.listenTo(this.model, 'change:atom_3d', this.addAtom)
        this.listenTo(this.model, 'change:field_idx', this.addField)
        this.listenTo(this.model, 'change:field_show', this.addField)
        this.listenTo(this.model, 'change:field_idx', this.addContour)
        this.listenTo(this.model, 'change:cont_show', this.addContour)
        this.listenTo(this.model, 'change:cont_axis', this.addContour)
        this.listenTo(this.model, 'change:cont_num', this.addContour)
        this.listenTo(this.model, 'change:cont_lim', this.addContour)
        this.listenTo(this.model, 'change:cont_val', this.addContour)
        this.listenTo(this.model, 'change:atom_3d', this.addAxis)
        this.listenTo(this.model, 'change:axis', this.addAxis)
        this.listenTo(this.model, 'change:tens', this.addTensor)
        this.listenTo(this.model, 'change:scale', this.addTensor)
        this.listenTo(this.model, 'change:tidx', this.colorTensor)
        this.listenTo(this.model, 'change:fill_idx', this.addAtom)
        this.listenTo(this.model, 'change:bond_r', this.addAtom)
        this.listenTo(this.model, 'change:clear_selected', this.clearSelected)
        this.listenTo(this.model, "change:freq", this.visualize_freq);
        this.listenTo(this.model, "change:freq_idx", this.visualize_freq);
        this.listenTo(this.model, "change:freq_scale", this.visualize_freq);
    }
}
