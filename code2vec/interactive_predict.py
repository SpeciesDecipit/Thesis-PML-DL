import glob

from common import common
from extractor import Extractor

SHOW_TOP_CONTEXTS = 10
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
JAR_PATH = '../code2vec/JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar'


class InteractivePredictor:
    exit_keywords = ['exit', 'quit', 'q']

    def __init__(self, config, model):
        model.predict([])
        self.model = model
        self.config = config
        self.path_extractor = Extractor(
            config,
            jar_path=JAR_PATH,
            max_path_length=MAX_PATH_LENGTH,
            max_path_width=MAX_PATH_WIDTH,
        )

    def read_file(self, input_filename):
        with open(input_filename, 'r') as file:
            return file.readlines()

    def predict(self, target_source_code, target_source_code_embeddings_output):
        input_folder = target_source_code
        for input_filename in glob.glob(f'{input_folder}/*.*'):
            predict_lines, hash_to_string_dict = self.path_extractor.extract_paths(input_filename)
            raw_prediction_results = self.model.predict(predict_lines)
            method_prediction_results = common.parse_prediction_results(
                raw_prediction_results,
                hash_to_string_dict,
                self.model.vocabs.target_vocab.special_words,
                topk=SHOW_TOP_CONTEXTS,
            )
            for raw_prediction, method_prediction in zip(
                raw_prediction_results, method_prediction_results
            ):
                if self.config.EXPORT_CODE_VECTORS:
                    vector_str = ' '.join(map(str, raw_prediction.code_vector))
                    output_path = f'{target_source_code_embeddings_output}/{input_filename.split("/")[-1].split(".")[0]}.txt'
                    open(output_path, 'a',).write(
                        vector_str + '\n',
                    )
