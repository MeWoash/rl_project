from dm_control import mujoco
import matplotlib.pyplot as plt


if __name__ == "__main__":
    model = mujoco.Physics.from_xml_path("D:/kody/rl_project/out/mjcf/out.xml")
    img = model.render(720, 1280, camera_id=0, overlays=[mujoco.TextOverlay("testTitle", "testBody","normal", "top")])
    plt.imshow(img)
    plt.show()