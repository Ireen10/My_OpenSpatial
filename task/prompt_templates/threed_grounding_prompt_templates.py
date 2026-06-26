object_grounding_box_template_questions = [
    "Identify the 3D bounding box surrounding the [A] within this environment.",
    "Locate the 3D bounding volume for the [A] present in the scene.",
    "Find the 3D bounding box that encapsulates the [A] in this visual representation.",
    "Extract the 3D bounding box coordinates of the [A] located in the image.",
    "Outline the 3D bounding box for the [A] visible in this setting.",
    "Pinpoint the 3D bounding box enclosing the [A] in this layout.",
    "Trace the edges of the 3D bounding box around the [A] in this scenario.",
    "Highlight the 3D bounding box that frames the [A] observed in the image.",
    "Predict the 3D location of the [A] observed in the image.",
]

grounding_camera_introduction = [
    (
        "Here are the detailed camera parameters for the image. "
        "Camera intrinsic parameters: Focal length f_x=[FX], f_y=[FY]. "
        "The principal point is near the center of the image: "
        "c_x=[CX] and c_y=[CY], with image width [W] and height [H]. "
        "We do not consider distortion parameters here. "
        "Therefore, the intrinsic matrix K = [[[FX], 0, [CX]], [0, [FY], [CY]], [0, 0, 1]]. "
        "Camera coordinates are defined as follows: the X-axis points rightward, the Y-axis points downward, "
        "and Z-axis points forward. The origin point is the camera location. "
        "We use the camera coordinate system as the world coordinate system; "
        "therefore, the camera extrinsic matrix is [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]."
    ),
]

grounding_json_output_instruction = [
    (
        'Output a json list in the following format: [{"label": "label_name", '
        '"bbox_3d": [x_center,y_center,z_center,x_size,y_size,z_size,roll,pitch,yaw]}]. '
        "Note: (1) label_name: the object name or its referring expression (optional). "
        "(2) x_center, y_center, z_center: the center of the object in the camera coordinate, in meters. "
        "(3) x_size, y_size, z_size: the dimensions of the object along the XYZ axes, in meters, "
        "when the rotation angles are zero. "
        "(4) roll, pitch, yaw: Euler angles representing rotations around the X, Y, and Z axes, "
        "respectively, expressed in radians ranging from -Pi to Pi."
    ),
]
from ..annotation.core.structured_prompt_template import AnswerInstructionProfile
from .register_structured import register_oe


def register_structured_grounding_templates() -> None:
    register_oe(
        "grounding_3d.open_ended",
        object_grounding_box_template_questions,
        [],
        introduction=grounding_camera_introduction,
        question_instruction=grounding_json_output_instruction,
        answer_profiles={
            "direct": AnswerInstructionProfile("direct", answer_templates=["[X]"]),
        },
        instruction_types=["direct"],
    )


register_structured_grounding_templates()
