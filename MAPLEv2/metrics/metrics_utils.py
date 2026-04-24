"""
Accuracy metrics for benchmark tasks.

- Closed-set information retrieval: 
    - Precision
    - Recall
    - F1
- Open-set information retrieval:
    - Character/word edit distance
    - Judge-LLM matching
- Numerical comparisons: 
    - Mean relative error
    - Time difference
    - Geographic distance


Author: Ian Stewart
Modified by: Rounak Meyur
"""

from typing import List, Union, Union, Dict, Any
import nltk
from nltk.metrics import edit_distance
from nltk.tokenize import WordPunctTokenizer
import numpy as np
from dateutil import parser
from geopy.distance import great_circle
from datetime import timedelta
from time import time
import os
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.chat_engine import SimpleChatEngine
from ast import literal_eval
import re
import tqdm
## download tokenizer model

if __name__ == "__main__":
    try:
        nltk.download('punkt_tab')
    except Exception as e:
        pass
    ## constants
    TOKENIZER = WordPunctTokenizer()
    MAX_DIST = 24901.461 / 2.0 # 1/2 of Earth circumference miles = max error
    ## max time error
    MAX_YEAR_DIFF = 100
    MAX_SECOND_DIFF = 60 * 60 * 24 * 365.25 * MAX_YEAR_DIFF

def closed_set_metrics(true_labels: List[str], pred_labels: List[str]):
    """
    Generate precision, recall, and F1 for predicted
    labels vs. true labels.

    Args:
        true_labels (List[str]): True labels for data
        pred_labels (List[str]): Predicted labels for data
    
    Returns:
        prec (float): Precision
        rec (float): Recall
        f1 (float): F1 score
    """
    tp = len(set(true_labels) & set(pred_labels))
    fp = len(set(pred_labels) - set(true_labels))
    fn = len(set(true_labels) - set(pred_labels))
    if(tp + fp > 0):
        prec = tp / (tp + fp)
    else:
        prec = 0.
    rec = tp / (tp + fn)
    f1 = (prec + rec) / 2.0
    return prec, rec, f1

class EditDistanceEvaluator():
    """
    Edit distance evaluator.
    """

    def __init__(self, tokenizer=None):
        """
        Initialize edit distance evaluator.

        Args:
            **kwargs: Additional parameters for specific implementations.
        """
        self.name = 'edit_distance_metric'
        self.tokenizer = tokenizer

    def edit_dist_metric(self, true_labels: List[str], pred_labels):
        """
        Compute string edit distance of best matches from true labels
        to predicted labels.

        Args:
            true_labels (List[str]): True labels for data
            pred_labels (List[str]): Predicted labels for data
        
        Returns:
            mean_edit_dist (float): Mean edit distance between predicted labels and best-matching true labels.
        """
        pred_matches = []
        ## if tokenizer provided, compute word edit distance
        if(self.tokenizer is not None):
            true_labels = list(map(self.tokenizer.tokenize, true_labels))
            pred_labels = list(map(self.tokenizer.tokenize, pred_labels))
        for pred_value in pred_labels:
            pred_label_len = len(pred_value)
            edit_dists = np.array(list(map(lambda x: edit_distance(pred_value, x) / max(pred_label_len, len(x)), true_labels)))
            min_edit_dist = np.min(edit_dists)
            best_match = pred_labels[np.argmin(edit_dists)]
            pred_matches.append((min_edit_dist, best_match))
        mean_edit_dist = np.mean(list(map(lambda x: x[0], pred_matches)))
        return mean_edit_dist

    def evaluate(self, ground_truth: Any, predicted: Any) -> Dict[str, Any]:
        """
        Evaluate the predicted answer against the ground truth.

        Args:
            ground_truth (Any): The ground truth/expected answer.
            predicted (Any): The predicted/generated answer.
            **kwargs: Additional parameters for evaluation.

        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics.
        """
        edit_dist = 0.
        edit_dist = self.edit_dist_metric(ground_truth, predicted)
        metric_dict = {'edit-dist' : edit_dist}
        return metric_dict

    def get_metric_names(self) -> List[str]:
        """
        Returns the list of metric names provided by this evaluator.

        Returns:
            List[str]: List of metric names.
        """
        return self.name

def convert_to_number(num_str, try_date=True):
    """
    Convert string to float or array.

    Args:
        num_str (str): String representing one/more number(s)
    
    Returns:
        num (float, numpy.array): Number
    """
    num = None
    if(try_date):
        ## try date
        try:
            num = parser.parse(num_str, fuzzy=True)
        except Exception as e:
            pass
    ## try floats
    if(num is None):
        num_vals = re.findall(r'([-\d\.]+)', num_str)
        num_vals = list(map(float, num_vals))
        if(len(num_vals) > 1):
            num = np.array(num_vals)
        else:
            num = num_vals[0]
    return num

def lat_lon_dist(lat_lon_1, lat_lon_2):
    """
    Compute relative distance [0,1] between latitude/longitude coordinates.

    Args:
        lat_lon_1 (List[List[float]]): Coordinate 1
        lat_lon_2 (List[List[float]]): Coordinate 2
    
    Returns:
        geo_dist (float): Great circle distance between coordinates
    """
    MAX_DIST = 24901.461 / 2.0 # 1/2 of Earth circumference miles = max error
    geo_err = great_circle(lat_lon_1, lat_lon_2)
    geo_err = geo_err.miles
    geo_err = geo_err / MAX_DIST
    return geo_err

def num_metric(
    true_value : Union[float, np.ndarray], 
    pred_value : Union[float, np.ndarray], 
    is_lat_lon : bool = False, 
    is_date_value = False,
    num_bounds : List[float] = None
):
    """
    Compute numerical error metric between true label and
    predicted label.

    Args:
        true_value (float | numpy.array): True value
        pred_value (float | numpy.array): Predicted value
        lat_lon (bool): Whether values are latitude/longitude coordinates
        num_bounds (List[float]): Min/max bounds for number
    
    Return:
        err (float): [0,1] numerical error
    """
    is_date_value = not is_lat_lon and is_date_value
    if(type(true_value) is str):
        true_label_num = convert_to_number(true_value, is_date_value)
    else:
        true_label_num = true_value
    if(type(pred_value) is str):
        pred_label_num = convert_to_number(pred_value, is_date_value)
    else:
        pred_label_num = pred_value
    ## lat/lon data = great circle distance
    if(is_lat_lon):
        err = lat_lon_dist(true_label_num, pred_label_num)
    ## metric = normalized error
    else:
        err = true_label_num - pred_label_num
        ## if array, compute mean squared err
        if(type(err) is np.ndarray):
            err = np.sqrt((err ** 2.0).sum())
        ## normalize date vals
        ## err % = seconds / seconds
        elif(type(err) is timedelta):
            ## max time error
            MAX_YEAR_DIFF = 100
            MAX_SECOND_DIFF = 60 * 60 * 24 * 365.25 * MAX_YEAR_DIFF
            err = err.total_seconds() / MAX_SECOND_DIFF
        ## bounded error
        elif(num_bounds is not None):
            err = abs(err - num_bounds[0]) / (num_bounds[1] - num_bounds[0])
        ## default err = absolute percent error
        else:
            err = abs(err) / abs(true_label_num)
    ## cap error at [0.0, 1.0]
    err = max(min(err, 1.0), 0.)
    ## score = 1 - err
    # score = 1 - err
    return err

def convert_str_to_list(str_data : str):
    """
    Convert string to list.

    Args:
        str_data (str): Data as string

    Returns:
        data_list (List[str]): Data as list
    """
    data_list = list(map(lambda x: x.strip(), str_data.split(',')))
    return data_list

## LLM methods
class AppV2OpenAIHandler():
    """Handler for loading Azure models"""
    def __init__(self, model_name="gpt-4o"):
        self.model_name=model_name

    def get_external_credentials(self):

        client_kwargs = {}
        client_kwargs["azure_openai_endpoint"]=os.environ.get("AZURE_OPENAI_ENDPOINT", "https://your-azure-openai-resource.openai.azure.com/")
        client_kwargs["azure_openai_api_key"]=os.environ.get("AZURE_OPENAI_API_KEY", None)
        client_kwargs["azure_openai_api_version"]=os.environ.get("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
        client_kwargs["azure_deployment_name"]=os.environ.get("AZURE_DEPLOYMENT_NAME", self.model_name)
    
        return client_kwargs

    def get_llm(self, client_kwargs):
        llm = AzureOpenAI(
            model=self.model_name,
            engine=client_kwargs["azure_deployment_name"],
            temperature=0.0,
            azure_endpoint=client_kwargs["azure_openai_endpoint"],
            api_key=client_kwargs["azure_openai_api_key"],
            api_version=client_kwargs["azure_openai_api_version"],
        )
        return llm 

class JudgeLLMEvaluator:

    def __init__(self, model_name):
        self.metric_name = 'judge-llm'
        ## default label match template
        self.label_match_prompt_template = """
        Determine if the source label is an approximate match for one or more of the target labels.
        The answer should be in JSON format and should reference which of the target labels is matched to the source label.

        Example source label:
        "dog"

        Example target labels:
        ["cat", "big dog"]

        Example match target labels:
        {{'match_target_labels' : ["big dog"]}}

        Source label:
        {}

        Target labels:
        {}

        Match target labels:
        """
        # match_key : JSON data key to extract from LLM output
        self.match_key = 'match_target_labels'
        ## NOTE: assume AZURE_OPENAI_API_KEY already defined before running
        ## NOTE: may fail due to stochastic model output
        appHandler = AppV2OpenAIHandler(model_name)
        client_kwargs = appHandler.get_external_credentials()
        llm = appHandler.get_llm(client_kwargs)
        self.model = SimpleChatEngine.from_defaults(llm=llm)

    def get_llm_matches(self, data, true_var, pred_var):
        """
        Extract approximate-match labels from LLM output and 
        add to existing data.

        Args:
            data (pandas.DataFrame): Dataframe, row = per-document extracted data
            true_var (str): Variable name for true label
            pred_var (str): Variable name for predicted label
            model (SimpleChatEngine): Chat Engine to submit prompts to LLM
            
        Returns:
            pred_label_matches (List[List[str]]): True label matches to predicted labels
        """
        pred_label_matches = []
        for idx, row in tqdm(data.iterrows()):
            true_labels = [row.loc[true_var]]
            pred_value = row.loc[pred_var]
            judge_response = self.prompt_llm(
                true_labels, pred_value,
                self.model,
                self.label_match_prompt_template
            )
            response_data = self.extract_data_from_str(judge_response)
            if(len(response_data) > 0):
                pred_label_matches.append(response_data['match_target_labels'])
            else:
                pred_label_matches.append([])
        return pred_label_matches
    
    def prompt_llm(
        self,
        true_labels : List[str], 
        pred_label : str, 
        model : SimpleChatEngine, 
        prompt_template : str
    ):
        """
        Prompt LLM to judge match between true labels and predicted labels.

        Args:
            true_labels (List[str]): True labels
            pred_label (str): Predicted label
            model (SimpleChatEngine): Chat Engine to submit prompts to LLM
            prompt_template (str): Prompt template, to contain true/predicted labels

        Returns:
            txt (str): Model response
        """
        prompt = prompt_template.format(pred_label, '\n'.join(true_labels))
        response = model.stream_chat(
            prompt
        )
        ## generate response per-token
        for token in response.response_gen:
            pass
        txt = response.response
        return txt

    def extract_data_from_str(self, output_str : str, json_matcher : str):
        """
        Extract JSON data from raw string.

        Args:
            output_str (str) : Output string containing JSON data.
            json_matcher (str): JSON-matching string pattern.
        
        Returns:
            output_data (dict) : JSON data
        """
        output_data_str = re.search(json_matcher, re.sub('\n', '', output_str))
        if(output_data_str is not None):
            output_data_str = output_data_str.group(0)
            ## parse
            output_data = literal_eval(output_data_str)
        return output_data

    def extract_llm_match(
        self,
        true_labels : List[str], 
        pred_label : str
    ):
        """
        Prompt LLM for approximate-match labels and parse response.

        Args:
            true_labels (List[str]): True labels
            pred_label (str): Predicted label
            model (SimpleChatEngine): Chat Engine to submit prompts to LLM
        
        Returns:
            match_data (List[str]): Matching true labels
        """
        if(type(true_labels) is str):
            true_labels = [true_labels]
        judge_response = self.prompt_llm(
            true_labels, pred_label,
            self.model,
            self.label_match_prompt_template
        )
        ## match specific key within output
        json_matcher = r'\{[ "]*' + self.match_key + r'"[ ]*:[^\}]+\}'
        response_data = self.extract_data_from_str(judge_response, json_matcher)
        match_data = []
        if(len(response_data) > 0):
            match_data = response_data[self.match_key]
        return match_data

    def evaluate(self, ground_truth: Any, predicted: Any, **kwargs) -> Dict[str, Any]:
        """
        Evaluate the predicted answer against the ground truth.

        Args:
            ground_truth (Any): The ground truth/expected answer.
            predicted (Any): The predicted/generated answer.
            **kwargs: Additional parameters for evaluation.

        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics.
        """
        if(type(ground_truth) is not list):
            ground_truth = [ground_truth]
        match_data = self.extract_llm_match(ground_truth, predicted)
        metric_val = 0
        if(len(match_data) > 0):
            metric_val = 1
        metric_dict = {'is-llm-match' : metric_val, 'llm-matches' : match_data}
        return metric_dict

    def get_metric_names(self) -> List[str]:
        """
        Returns the list of metric names provided by this evaluator.

        Returns:
            List[str]: List of metric names.
        """
        return self.metric_name

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
class SemanticScoreEvaluator:

    def __init__(self, model_name_or_path):
        self.sentence_encoder = SentenceTransformer(model_name_or_path)
        self.metric_name = 'semantic-match'

    def get_sims_best_matches(self, true_labels, pred_labels):
        """
        Compute similarity scores for best matches.
        """
        true_label_emb = self.sentence_encoder.encode(true_labels)
        pred_label_emb = self.sentence_encoder.encode(pred_labels)
        sims = cosine_similarity(
            pred_label_emb, true_label_emb
        )
        ## set min = 0.
        sims = np.maximum(sims, 0.)
        ## get max sim
        max_sims = np.max(sims, axis=1)
        max_sim_idx = np.argmax(sims, axis=1)
        max_sim_matches = [true_labels[i] for i in max_sim_idx]
        return max_sims, max_sim_matches

    def evaluate(self, ground_truth: Any, predicted: Any,  match_threshold=0.75) -> Dict[str, Any]:
        best_match_scores, _ = self.get_sims_best_matches(ground_truth, predicted)
        mean_match_score = np.mean(best_match_scores)
        is_match = mean_match_score >= match_threshold
        metric_dict = {
            'mean-match-similarity' : mean_match_score, 
            'match-scores' : best_match_scores,
            'is-match' : is_match,
        }
        return metric_dict

## unit tests
def test_closed_set_metrics():
    """
    Test closed set metrics.
    """
    true_labels = ['Ohio', 'California', 'Florida', 'Maine']
    pred_labels = ['Ohio', 'Texas', 'Florida', 'Pennsylvania', 'Oregon']
    prec, rec, f1 = closed_set_metrics(true_labels, pred_labels)
    assert prec == 0.4
    assert rec == 0.5
    assert f1 == 0.45

def test_num_metrics():
    """
    Test numeric metrics:

    - float
    - array
    - date
    - latitude/longitude
    """
    ## float
    true_num = '0.5'
    pred_num = '0.3'
    err = num_metric(true_num, pred_num)
    true_err = 0.4
    assert err == true_err
    ## array
    true_num = '0.0, 1.0'
    pred_num = '0.5, 0.5'
    err = num_metric(true_num, pred_num)
    true_err = (0.5**2 + 0.5**2)**0.5
    assert err == true_err
    ## date
    ## max time error
    MAX_YEAR_DIFF = 100
    MAX_SECOND_DIFF = 60 * 60 * 24 * 365.25 * MAX_YEAR_DIFF
    true_num = 'March 1 2025'
    pred_num = 'February 1 2025'
    err = num_metric(true_num, pred_num, is_date_value=True)
    true_err = 28*24*60*60 / MAX_SECOND_DIFF
    assert err == true_err
    ## lat/lon
    MAX_DIST = 24901.461 / 2.0 # 1/2 of Earth circumference miles = max error
    true_num = '10.0, 10.0'
    pred_num = '-5.0, -5.0'
    err = num_metric(true_num, pred_num, is_lat_lon=True)
    true_err = great_circle((10.0, 10.0), (-5.0, -5.0)).miles / MAX_DIST
    assert err == true_err


def test_open_set_metrics():
    """
    Test open-set metrics:

    - char edit distance
    - word edit distance
    - judge LLM
    - semantic similarity
    """
    # edit distance: char
    edit_dist_evaluator = EditDistanceEvaluator(tokenizer=None)
    true_labels = ['California']
    pred_labels = ['Cali']
    err = edit_dist_evaluator.evaluate(true_labels, pred_labels)
    true_err = 6 / 10
    assert true_err == err['edit-dist']
    # edit distance: word
    tokenizer = WordPunctTokenizer()
    edit_dist_evaluator = EditDistanceEvaluator(tokenizer=tokenizer)
    true_labels = ['New York']
    pred_labels = ['New York City']
    err = edit_dist_evaluator.evaluate(true_labels, pred_labels)
    true_err = 1 / 3
    assert true_err == err['edit-dist']
    ## semantic distance
    model_name_or_path = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    semantic_sim_evaluator = SemanticScoreEvaluator(model_name_or_path)
    true_labels = ['Native Village of Tetlin']
    pred_labels = ['native village tetlin']
    true_sim = 0.9882463
    match_threshold = 0.75
    sim = semantic_sim_evaluator.evaluate(true_labels, pred_labels, match_threshold=match_threshold)
    assert sim['mean-match-similarity'] == true_sim
    assert sim['is-match']
    ## LLM
    model_name='gpt-4o'
    judge_llm_metric = JudgeLLMEvaluator(model_name)
    true_labels = ['Colorado']
    ## positive match
    pred_label = 'CO'
    pred_match = judge_llm_metric.evaluate(true_labels, pred_label)
    assert pred_match['is-llm-match'] == 1
    ## negative match
    pred_label = 'WI'
    pred_match = judge_llm_metric.evaluate(true_labels, pred_label)
    assert pred_match['is-llm-match'] == 0

def main():
    # test_closed_set_metrics()
    # test_num_metrics()
    test_open_set_metrics()

if __name__ == '__main__':
    main()
