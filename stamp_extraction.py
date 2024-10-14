import ast
from mmengine.logging import print_log
from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes

def process_image(inputs, model, weights=None, out_dir='outputs', texts=None, device='cuda', 
                  pred_score_thr=0.3, batch_size=1, show=False, no_save_vis=False, 
                  no_save_pred=False, print_result=False, palette='none', 
                  custom_entities=False, chunked_size=-1, tokens_positive=None):
    """
    Process an input image or folder using a specified model.
    
    Args:
        inputs (str): Input image file or folder path.
        model (str): Model configuration or checkpoint file.
        weights (str, optional): Path to checkpoint file. Defaults to None.
        out_dir (str, optional): Output directory for images or prediction results. Defaults to 'outputs'.
        texts (str, optional): Text prompt for detection. Defaults to None.
        device (str, optional): Device used for inference. Defaults to 'cuda'.
        pred_score_thr (float, optional): BBox score threshold. Defaults to 0.3.
        batch_size (int, optional): Inference batch size. Defaults to 1.
        show (bool, optional): Whether to display the image. Defaults to False.
        no_save_vis (bool, optional): Do not save visualization results. Defaults to False.
        no_save_pred (bool, optional): Do not save prediction results. Defaults to False.
        print_result (bool, optional): Whether to print results. Defaults to False.
        palette (str, optional): Color palette for visualization. Defaults to 'none'.
        custom_entities (bool, optional): Customize entity names. Defaults to False.
        chunked_size (int, optional): Chunked size for large categories. Defaults to -1.
        tokens_positive (str, optional): Specified interest locations in input text. Defaults to None.
    
    Returns:
        
    """
    if no_save_vis and no_save_pred:
        out_dir = ''

    if model.endswith('.pth'):
        print_log('The model is a weight file, automatically assigning the model to --weights')
        weights = model
        model = None
    print(weights)
    print(model)
    if model is None and weights is None:
        raise ValueError("Either 'model' or 'weights' must be specified.")
    if texts is not None and texts.startswith('$:'):
        dataset_name = texts[3:].strip()
        class_names = get_classes(dataset_name)
        texts = [tuple(class_names)]

    if tokens_positive is not None:
        tokens_positive = ast.literal_eval(tokens_positive)
    print("Local variables:", locals())

    #init_kws = ['model', 'weights', 'device', 'palette']
    #init_args = {kw: locals()[kw] for kw in init_kws if kw in locals()}  # Ensure that we only access existing local variables
    init_args = {
        'model': model,
        'weights': weights,
        'device': 'cpu',
        'palette': palette
    }
    
    print("Initialization arguments:", init_args)  # Debug print for initialization arguments
    print(init_args)
    # Initialize the inference model
    inferencer = DetInferencer(**init_args)
    chunked_size = locals().get('chunked_size', -1)
    inferencer.model.test_cfg.chunked_size = chunked_size

    # Process the inputs using the inferencer
    call_args ={
    'inputs': inputs, 
    'out_dir': out_dir, 
    'texts':texts, 
    'pred_score_thr': pred_score_thr, 
    'batch_size': 1, 
    'show': show, 
    'no_save_vis': no_save_vis,
     'no_save_pred': no_save_pred, 
     'print_result': print_result, 
     'custom_entities': custom_entities, 
     'tokens_positive': None}
    inferencer(**call_args)

    # if out_dir != '' and not (no_save_vis and no_save_pred):
    #     print_log(f'Results have been saved at {out_dir}')
    #     return f'Results have been saved at {out_dir}'

    # return "Processing completed successfully."
