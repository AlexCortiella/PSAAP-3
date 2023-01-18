#MAIN TRAIN

def main(cfg):
    
    
    predictor = Predictor(cfg, './results/logs/run_0/snapshot.pt')
    print('Prediction initialized...')
    predictions = predictor.predict()
    print('Prediction completed!')
    return predictions


if __name__ == "__main__":
    
    import argparse
    import pprint
    import yaml
    from core.model import *
    from core.data import *
    from core.predictor import *
    from core.utils import *
    
    parser = argparse.ArgumentParser(description='Input configuration file')
    parser.add_argument('-f','--config', type=str,
                          help='Configuration filepath that contains model parameters',
                          default = './config_file_multigpu.yaml')
    parser.add_argument('-j','--jobid', type=str,
                          help='JOB ID',
                          default = '000000')

    args = parser.parse_args()

    config_filepath = args.config

    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = dic2struc(cfg)
    
    predictions = main(cfg)
    
    pred_path = os.path.join(cfg.predictions_path, "predictions.json")
    
    print('Saving predictions...')
    with open(pred_path, "w") as outfile:
        json.dump(predictions, outfile)
    
    
