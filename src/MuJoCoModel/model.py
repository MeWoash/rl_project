import mujoco
import mujoco.viewer
import typing

import sys, os

XML_FOLDER = os.path.normpath(os.path.join(__file__,"..","xmls"))
MODEL_PATH = os.path.join(XML_FOLDER, "test1.xml")



if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)

    mujoco.viewer.launch(model) #  launches simulate.exe to debug simulation