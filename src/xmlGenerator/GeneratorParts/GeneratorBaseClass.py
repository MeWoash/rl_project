import abc
import xml.etree.ElementTree as ET
from typing import Any, Dict


class BaseGenerator(abc.ABC):
    def __init__(self, props: dict = {}) -> None:
        super().__init__()
        self.props: dict = props

    @abc.abstractmethod
    def attachToMujoco(self, mujocoNode: ET.Element) -> None:
        """
        Function which defines how derived class is connected with main MuJoCo Tree.

        Args:
            mujocoNode (ET.ElementTree): main MuJoCo tree

        Raises:
            NotImplementedError: If not defined in derived class.

        Returns:
            None:
        """
        raise NotImplementedError("Implement attachToMujoco method")

    @abc.abstractmethod
    def _calculateProperties(self) -> None:
        """
        Every derived class should use it to calculate internal variables.

        Raises:
            NotImplementedError: If not defined in derived class.

        Returns:
            None: 
        """
        raise NotImplementedError("Implement _calculateProperties method")

    @abc.abstractmethod
    def generateNodes(self) -> Dict[str, ET.Element]:
        """
        Generates all TEMPLATES elements with values in self.props of derived class

        Raises:
            NotImplementedError: If TEMPLATES is not defined in derived class

        Returns:
            dict: dictionary of generated nodes.
        """
        templates = getattr(self, 'TEMPLATES', None)
        if templates is None:
            raise NotImplementedError("Add TEMPLATES var")
        nodeDict: dict = {}
        if templates is not None:
            for template_name, template_string in templates.items():
                replaced = template_string.format(**self.props)
                node: ET.Element = ET.fromstring(replaced)
                nodeDict[template_name] = node
        return nodeDict
