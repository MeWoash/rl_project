import abc
import xml.etree.ElementTree as ET
from typing import Any, Dict


class BaseGenerator(abc.ABC):
    def __init__(self, props: dict = {}) -> None:
        super().__init__()
        self.props: dict = props

    @abc.abstractmethod
    def attachToMujoco(self, mujocoNode: ET.Element) -> Any:
        raise NotImplementedError("Implement attachToMujoco method")

    def generateNodes(self) -> list:
        templates = getattr(self, 'TEMPLATES', None)
        if templates is None:
            raise NotImplementedError("Add TEMPLATES var")
        nodeList = []
        if templates is not None:
            for template_name, template_string in templates.items():
                replaced = template_string.format(**self.props)
                node: ET.Element = ET.fromstring(replaced)
                nodeList.append(node)
        return nodeList
