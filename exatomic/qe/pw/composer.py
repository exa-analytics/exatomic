
class PWInputMeta(ComposerMeta):
    """
    """
    control = dict
    system = dict
    electrons = dict
    ions = dict
    cell = dict


class PWInput(six.with_metaclass(PWInputMeta, PWCPComposer)):
    """
    Input file composer for Quantum ESPRESSO's pw.x module.
    """
    _template = _pwtemplate
