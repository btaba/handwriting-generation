from handwriting_gen.data_manager import get_model_dir
from handwriting_gen.unconditional_stroke_model import (
    UnconditionalStrokeModel,
    decode as unconditional_decode
)


MODEL_DIR = get_model_dir()


def generate_unconditionally(random_seed=1):
    """
    Input:
      random_seed - integer

    Output:
      stroke - numpy 2D-array (T x 3)
    """
    model = UnconditionalStrokeModel.load(
        str(MODEL_DIR / 'unconditional-stroke-model2'),
        batch_size=1, rnn_steps=1, is_train=False)
    return unconditional_decode(model, seed=random_seed)


def generate_conditionally(text='welcome', random_seed=1):
    """
    Input:
      text - str
      random_seed - integer

    Output:
      stroke - numpy 2D-array (T x 3)
    """
    pass


# def recognize_stroke(stroke):
#     # Input:
#     #   stroke - numpy 2D-array (T x 3)

#     # Output:
#     #   text - str
#     return 'welcome'
