__author__ = "Dong Yihan"

import json
import ast
import math
import os
import numpy as np
from .language_model import OPENAI


class PostProcess:
    """
    議論の発言から主張の抽出、queryにの変換、及び各主張と元の投稿の確信度の計算のため、PostProcess（投稿処理）というクラスです。
    """

    def __init__(self):
        """
        投稿処理クラスのオブジェクトの初期化メソッド。
        """
        self.status = {
            "status": 500,
            "message": "error in initialization of PostProcess"
        }

        # 言語モデルを用い、タスクを果たすため、プロンプトの読み込み
        with open("./prompts/claim_extraction.json", "r") as claim_extraction_file:
            claim_data = json.load(claim_extraction_file)
        self.claim_extraction_instruct = {
            "system": claim_data["system"],
            "user": claim_data["user"]
        }

        with open("./prompts/query_generation.json", "r") as query_generation_file:
            query_data = json.load(query_generation_file)
        self.query_generation_instruct = {
            "system": query_data["system"],
            "user": query_data["user"]
        }

        # openaiオブジェクトの初期化
        self.openai = OPENAI()

        # マルチエージェントの判断結果をあらかじめ定義するリストです。
        self.factuality_labels = ["FALSE", "PARTLY", "TRUE"]

        self.status.update(status=200, message="successfully initialize PostProcess")
        return

    def convert_post_to_claim(self, post: str):
        """
        実験を行うため、データセットから読み込んだ主張をさらに主張の抽出を行わず、直接に主張のリストの形に変形するメソッド。
        :param post: 事実検証データセットから読み込んだ主張です。strとしてシステム全体に投げられる
        :return: claim_extraction_response claim_extraction()の返事と同じようなlist[dict{"claim": "claim"}]の形です。
        """
        self.status.update(status=500, message="error in convert_post_to_claim")

        claim_list = post.split(". ")

        claim_extraction_response = []
        for claim in claim_list:
            if claim:
                extracted_claim = {
                    "claim": claim
                }
                claim_extraction_response.append(extracted_claim)

        self.status.update(status=200, message="successfully convert post to claim")
        return claim_extraction_response

    # 投稿から主張を抽出するメソッドです。一つの投稿から抽出するためここのpostは一つのstrです。
    def claim_extraction(self, post: str):
        """
        投稿から主張を抽出するメソッドです。一つの投稿しか抽出できない。
        :param post: 元の投稿。str
        :return: claim_extraction_responses 抽出された主張のリスト。list[dict{"claim": str}]
        """
        self.status.update(status=500, message="error in claim_extraction")

        claim_extraction_prompts = [
            {
                "role": "system",
                "content": self.claim_extraction_instruct["system"]
            },
            {
                "role": "user",
                "content": self.claim_extraction_instruct["user"].format(input=post)
            }
        ]

        claim_extraction_responses = self.openai.get_gpt_response(prompts=claim_extraction_prompts)
        claim_extraction_responses = ast.literal_eval(claim_extraction_responses)  # ここで得られる主張はlist[dict]という形です

        print(claim_extraction_responses)
        self.status.update(status=200, message="successfully get claims from a post")

        return claim_extraction_responses

    def query_generation(self, claim_list: list):
        """
        抽出された主張を疑問の形式に変換するメソッドです。
        :param claim_list: 抽出された主張のリストです。複数の主張が一つの投稿から抽出された可能性があるので
        これはlist[dict{"claim": str}]です。
        :return: query_generation_response = [{"query": "query1", "answer": "answer1"}, {}, ... , {}]
        """
        self.status.update(status=500, message="error in query_generation")

        if claim_list is None:
            self.status.update(status=500, message="there is no claim")
            return ['None']
        query_generation_responses = []
        for claim in claim_list:
            if "claim" in claim:
                query_generation_prompts = [
                    {
                        "role": "system",
                        "content": self.query_generation_instruct["system"]
                    },
                    {
                        "role": "user",
                        "content": self.query_generation_instruct["user"].format(input=claim["claim"])
                    }
                ]

                query_generation_response = self.openai.get_gpt_response(prompts=query_generation_prompts)
                print("query generation: \n", query_generation_response)
                query_generation_response = ast.literal_eval(query_generation_response)  # ここで得られる結果はdictです
                query_generation_responses.append(query_generation_response)
            else:
                continue

        '''
        query_generation_response = [{"query": "query1", "answer": "answer1"}, {}, ... , {}]
        '''
        print(query_generation_responses)
        self.status.update(status=200, message="successfully convert claims into queries")

        return query_generation_responses

    def confidence_normalization(self, confidence_upper_limit: float, confidence_lower_limit: float,
                                 confidence_real: float):
        """
        計算された全体的な確信度を正規化するメソッド。
        各主張の確信度が違いますので、最後に別べに計算された確信度と上限および下限の比較はややこしいです。
        正規化された確信度を最後の確信度として扱う。
        :parameter:
        confidence_upper_limit: float ある主張に対し、全部の証言がその主張に賛成する場合に、確信度の和です。+
        confidence_lower_limit: float ある主張に対し、全部の証言がその主張に反対する場合に、確信度の和です。-
        confidence_real: float ある主張の実の確信度。
        :return:
        confidence_normalized: 正規化された確信度
        """
        self.status.update(status=500, message="error in general_confidence_normalization")

        confidence_normalized = ((confidence_real - confidence_lower_limit) /
                                 (confidence_upper_limit - confidence_lower_limit))

        self.status.update(status=200, message="successfully normalize confidence of a claim")
        return confidence_normalized

    def calculate_sub_confidence(self, agent_results_list, total_agents_number=3):
        """
        ある主張に対し、その主張の確信度を計算するメソッド。
        :type agent_results_list: list
        エージェントで得た判断結果は全部list[dict{"claim": str, "confidence": float, "factuality": boolean}]です
        :param total_agents_number: エージェントの数
        :param agent_results_list: 各エージェントの判断結果の二次元リスト
        list[dict{"claim": str, "confidence": float, "factuality": boolean}]
        :return: claims_and_sub_confidence: list[dict{"claim": str, "sub_confidence": float, from 0 to 1}]
        """
        self.status.update(status=500, message="error in calculate_general_confidence")

        claims_and_sub_confidence = []
        length = len(agent_results_list[0])
        if any(len(lst) != length for lst in agent_results_list):
            print("check program! Not all the agent results have the same length")
            self.status.update(status=500, message="Not all the agent results have the same length")
            return claims_and_sub_confidence

        # 元の二次元リストの転置です。新しいリストには、要素ごとに同じ主張の信憑性結果が保存されている。
        # つまり、元のリストには、list[list[エージェント1の結果の集合], list[エージェント2の結果の集合]...]であり、
        # 新しいリストには、list[list[主張1に対し判断結果の集合], list[主張2に対し判断結果の集合]...]です。
        agent_results = np.array(agent_results_list).T.tolist()
        print(len(agent_results))
        # for agent_result in agent_results:
        #     for result in agent_result:
        #         result["confidence"] = float(result["confidence"])

        # 主張ごとの確信度を計算する
        for agent_result in agent_results:
            right_agents_number = len([result["factuality"] for result in agent_result
                                       if result["factuality"] is True])
            wrong_agents_number = total_agents_number - right_agents_number

            # 正解および誤りと判断したエージェントの結果の確信度の和

            right_agents_confidence_sum = sum([result["confidence"] for result in agent_result
                                               if result["confidence"] > 0])
            wrong_agents_confidence_sum = sum([result["confidence"] for result in agent_result
                                               if result["confidence"] < 0])

            # ある主張に対し、確信度の上限と下限を計算する
            # 確信度の上限は全ての証言をその主張に賛成する時に、公式で計算された結果であり
            # 確信度の下限は全ての証言をその主張に反対する時に、公式で計算された結果である。
            # 確信度の上限と下限を変わらない数値に変更する
            # confidence_upper_sum = sum([abs(result["confidence"]) for result in agent_result])
            # confidence_lower_sum = sum([-abs(result["confidence"]) for result in agent_result])
            confidence_upper_sum = len(agent_result) + 1
            confidence_lower_sum = -len(agent_result) - 1

            confidence_upper_limit = (math.log2(total_agents_number + 1) *
                                      (confidence_upper_sum / total_agents_number))
            confidence_upper_limit_square = (math.sqrt(total_agents_number + 1) *
                                             (confidence_upper_sum / total_agents_number))
            confidence_upper_limit_linear = ((total_agents_number + 1) *
                                             (confidence_upper_sum / total_agents_number))
            confidence_lower_limit = (math.log2(total_agents_number + 1) *
                                      (confidence_lower_sum / total_agents_number))
            confidence_lower_limit_square = (math.sqrt(total_agents_number + 1) *
                                             (confidence_lower_sum / total_agents_number))
            confidence_lower_limit_linear = ((total_agents_number + 1) *
                                             (confidence_lower_sum / total_agents_number))

            # まずは全体的な確信度を0と設定して、もしエージェントの数が0ではなければ公式に従って計算する
            # 12.22 簡単にスコアを計算して、提案手法と比べる
            # 提案手法
            if right_agents_number == 0:
                right_sub_confidence = 0
                right_sub_confidence_square = 0
                right_sub_confidence_linear = 0
            else:
                right_sub_confidence = (math.log2(right_agents_number + 1) *
                                        (right_agents_confidence_sum / right_agents_number))
                right_sub_confidence_square = (math.sqrt(right_agents_number + 1) *
                                               (right_agents_confidence_sum / right_agents_number))
                right_sub_confidence_linear = ((right_agents_number + 1) *
                                               (right_agents_confidence_sum / right_agents_number))
            if wrong_agents_number == 0:
                wrong_sub_confidence = 0
                wrong_sub_confidence_square = 0
                wrong_sub_confidence_linear = 0
            else:
                wrong_sub_confidence = (math.log2(wrong_agents_number + 1) *
                                        (wrong_agents_confidence_sum / wrong_agents_number))
                wrong_sub_confidence_square = (math.sqrt(wrong_agents_number + 1) *
                                               (wrong_agents_confidence_sum / wrong_agents_number))
                wrong_sub_confidence_linear = ((wrong_agents_number + 1) *
                                               (wrong_agents_confidence_sum / wrong_agents_number))

            # # 確信度の和
            # sum_confidence = right_agents_confidence_sum + wrong_agents_confidence_sum
            # sub_confidence = self.confidence_normalization(confidence_upper_limit=confidence_upper_sum,
            #                                                confidence_lower_limit=confidence_lower_sum,
            #                                                confidence_real=sum_confidence)

            # 平均値
            # confidence_average = (right_agents_confidence_sum + wrong_agents_confidence_sum) / 3
            # sub_confidence = confidence_average

            # 線形
            confidence_linear = right_sub_confidence_linear + wrong_sub_confidence_linear
            sub_confidence_linear = self.confidence_normalization(confidence_upper_limit=confidence_upper_limit_linear,
                                                                  confidence_lower_limit=confidence_lower_limit_linear,
                                                                  confidence_real=confidence_linear)

            # 平方根
            confidence_square = right_sub_confidence_square + wrong_sub_confidence_square
            sub_confidence_square = self.confidence_normalization(confidence_upper_limit=confidence_upper_limit_square,
                                                                  confidence_lower_limit=confidence_lower_limit_square,
                                                                  confidence_real=confidence_square)

            # # 提案手法
            confidence_real = right_sub_confidence + wrong_sub_confidence
            sub_confidence = self.confidence_normalization(confidence_upper_limit=confidence_upper_limit,
                                                           confidence_lower_limit=confidence_lower_limit,
                                                           confidence_real=confidence_real)

            # 簡単な手法
            # sub_confidence = self.confidence_normalization(confidence_upper_limit=confidence_upper_sum,
            #                                                confidence_lower_limit=confidence_lower_sum,
            #                                                confidence_real=confidence_real)

            claim_and_sub_confidence = {
                "claim": agent_result[0]["claim"],
                "sub_confidence": sub_confidence,
                "sub_confidence_square": sub_confidence_square,
                "sub_confidence_linear": sub_confidence_linear
            }
            claims_and_sub_confidence.append(claim_and_sub_confidence)

        self.status.update(status=200, message="successfully calculate general confidence of claims")
        return claims_and_sub_confidence

    # 投票（ウェイトなし）で主張の信憑性を獲得するため、各エージェントの結果により投票するメソッドの実装。
    def judge_original_post_factuality_voting_binary(self, original_post: str, claims_and_results_list):
        """
        MAFC Scoreの計算の代わりに、投票で元の主張の信憑性を確定するため、投票メソッドを実装します。

        このメソッドで獲得できる投稿の信憑性は二項（True/False）だけです。
        :param original_post: str 元の投稿です
        :param claims_and_results_list: 各エージェントからもらった主張、判断結果および結果の確信度が含めリストです。
        list[dict{"claim": str, "confidence": float, "factuality": boolean}]
        :return: post_and_confidence: dict{"post", "factuality"}
        元の投稿の信憑性は、投稿から抽出された主張の信憑性により決められる。
        主張ごとの信憑性は各エージェントからのもらった判断結果により決められる。
        つまり、post1からclaim1, claim2, claim3が抽出された場合には、claimごとの信憑性は投票の結果で判断され、
        post自体の信憑性はclaimsの信ぴょう性で判断される。
        """
        self.status.update(status=500, message="error in judge_original_claim")

        post_and_factuality = {}
        # google agentの判断結果の長さは標準として扱います。
        length = len(claims_and_results_list[0])  # 抽出された主張の数
        if any(len(lst) != length for lst in claims_and_results_list):
            print("check program! The length of results of each agent is not the same!")
            self.status.update(status=500, message="Not all the agents results have the same length")
            return post_and_factuality

        # 元の二次元リストの転置。新しいリストには、要素ごとに同じ主張の信憑性結果が保存されている。
        claims_and_results = np.array(claims_and_results_list).T.tolist()
        true_claims_number = 0

        # 各主張の信ぴょう性を投票でもらう
        for claim_and_result in claims_and_results:
            right_agents_number = len([result["factuality"] for result in claim_and_result
                                       if result["factuality"] is True])
            wrong_agents_number = len(claim_and_result) - right_agents_number
            if right_agents_number > wrong_agents_number:
                true_claims_number = true_claims_number + 1
        false_claim_number = len(claims_and_results) - true_claims_number

        # 正解の主張と誤りの主張の数を比べて、元の投稿の信ぴょう性を決める。
        if true_claims_number > false_claim_number:
            post_and_factuality = {
                "post": original_post,
                "factuality": "true"
            }
        else:
            post_and_factuality = {
                "post": original_post,
                "factuality": "false"
            }

        self.status.update(status=200, message="successfully get factuality of the original post by voting")
        return post_and_factuality

    def judge_original_post_factuality_voting_multilabel(self, original_post: str, claims_and_results_list):
        """
        MAFC Scoreの計算の代わりに、投票で元の主張の信憑性を確定するため、投票メソッドを実装します。

        このメソッドで獲得できる投稿の信憑性は複数のラベル（True/mostly true/half true/mostly false/False）を含める。
        :param original_post: str 元の投稿です
        :param claims_and_results_list: 各エージェントからもらった主張、判断結果および結果の確信度が含めリストです。
        list[dict{"claim": str, "confidence": float, "factuality": boolean}]
        :return: post_and_confidence: dict{"post", "factuality"}
        元の投稿の信憑性は、投稿から抽出された主張の信憑性により決められる。
        主張ごとの信憑性は各エージェントからのもらった判断結果により決められる。
        つまり、post1からclaim1, claim2, claim3が抽出された場合には、claimごとの信憑性は投票の結果で判断され、
        post自体の信憑性はclaimsの信ぴょう性で判断される。
        """
        self.status.update(status=500, message="error in judge_original_claim")

        post_and_factuality = {}
        # google agentの判断結果の長さは標準として扱います。
        length = len(claims_and_results_list[0])  # 抽出された主張の数
        if any(len(lst) != length for lst in claims_and_results_list):
            print("check program! The length of results of each agent is not the same!")
            self.status.update(status=500, message="Not all the agents results have the same length")
            return post_and_factuality

        # 元の二次元リストの転置。新しいリストには、要素ごとに同じ主張の信憑性結果が保存されている。
        claims_and_results = np.array(claims_and_results_list).T.tolist()
        print(claims_and_results)
        true_claims_number = 0
        right_agents_number = 0
        wrong_agents_number = 0

        # 各主張の信ぴょう性を投票でもらう
        for claim_and_result in claims_and_results:
            right_agents_number = right_agents_number + len([result["factuality"] for result in claim_and_result
                                                             if result["factuality"] is True])
            wrong_agents_number = wrong_agents_number + (len(claim_and_result) - right_agents_number)
            print("right agent number:", right_agents_number)
            print("wrong agent number:", wrong_agents_number)
            if right_agents_number > wrong_agents_number:
                true_claims_number = true_claims_number + 1
        false_claim_number = len(claims_and_results) - true_claims_number

        # 正解と判断された主張数によって、元の投稿の信憑性を決める。
        # 一つの主張の場合には、実は二項分類ですので、正しい主張の数ではなく正しいと判断したエージェント数で投稿の信憑性を決める。
        if wrong_agents_number == 0:
            post_and_factuality = {
                "post": original_post,
                "factuality": "true"
            }
        elif right_agents_number == 0:
            post_and_factuality = {
                "post": original_post,
                "factuality": "false"
            }
        else:
            post_and_factuality = {
                "post": original_post,
                "factuality": "half-true"
            }
        # if true_claims_number == len(claims_and_results):
        #     post_and_factuality = {
        #         "post": original_post,
        #         "factuality": "TRUE"
        #     }
        # elif false_claim_number == len(claims_and_results):
        #     post_and_factuality = {
        #         "post": original_post,
        #         "factuality": "FALSE"
        #     }
        # elif true_claims_number > false_claim_number:
        #     post_and_factuality = {
        #         "post": original_post,
        #         "factuality": "mostly true"
        #     }
        # elif true_claims_number == false_claim_number:
        #     post_and_factuality = {
        #         "post": original_post,
        #         "factuality": "partly true/misleading"
        #     }
        # elif true_claims_number < false_claim_number:
        #     post_and_factuality = {
        #         "post": original_post,
        #         "factuality": "mostly false"
        #     }

        self.status.update(status=200, message="successfully get factuality of the original post by voting")
        return post_and_factuality

    def judge_original_post_factuality(self, claims_and_sub_confidence, original_post: str):
        """
        最初の投稿の信憑性を計算するメソッドです。
        ユーザの投稿からいくつかの主張が抽出され、そちらの確信度を別々で計算され、最後に合わせて元の投稿の信憑性を測る。
        :param original_post: 元の投稿です。
        :param claims_and_sub_confidence: list[dict{"claim", "sub_confidence"}]
        claimは抽出された主張であり、sub_confidenceは正規化された主張の確信度である。0と近いのは誤り（あやまり）であり、1と近いのは事実である
        :return: post_and_confidence: dict{"post", "confidence", "factuality"}
        全体的な確信度は各主張の確信度の平均値です。
        """
        self.status.update(status=500, message="error in judge_original_claim_factuality")

        # for the ablation experiments
        confidence_comparison_save_dir = "./experiment_results/SciFact/confidence_comparison_multilabel_save_dir.txt"

        # claims_and_sub_confidenceからsub_confidenceだけを抽出する
        sub_confidence_list = [claim_and_sub_confidence["sub_confidence"] for
                               claim_and_sub_confidence in claims_and_sub_confidence]

        # for the ablation experiments
        sub_confidence_square_list = [claim_and_sub_confidence["sub_confidence_square"] for
                                      claim_and_sub_confidence in claims_and_sub_confidence]
        sub_confidence_linear_list = [claim_and_sub_confidence["sub_confidence_linear"] for
                                      claim_and_sub_confidence in claims_and_sub_confidence]

        # 元の投稿の全体的な確信度を計算する。0から1までの値です。
        # 切り捨て除算でラベルを加える
        factuality_labels = self.factuality_labels
        confidence_sum = sum(sub_confidence_list)
        confidence_sum_upper = len(sub_confidence_list)
        confidence_sum_lower = 0
        general_average_confidence = self.confidence_normalization(confidence_real=confidence_sum,
                                                                   confidence_upper_limit=confidence_sum_upper,
                                                                   confidence_lower_limit=confidence_sum_lower)

        # for the ablation experiments
        confidence_sum_square = sum(sub_confidence_square_list)
        general_average_confidence_square = self.confidence_normalization(confidence_real=confidence_sum_square,
                                                                          confidence_upper_limit=confidence_sum_upper,
                                                                          confidence_lower_limit=confidence_sum_lower)

        confidence_sum_linear = sum(sub_confidence_linear_list)
        general_average_confidence_linear = self.confidence_normalization(confidence_real=confidence_sum_linear,
                                                                          confidence_upper_limit=confidence_sum_upper,
                                                                          confidence_lower_limit=confidence_sum_lower)

        if general_average_confidence > 1:
            general_average_confidence = 1.0
        elif general_average_confidence < 0:
            general_average_confidence = 0.0

        if general_average_confidence_square > 1:
            general_average_confidence_square = 1.0
        elif general_average_confidence_square < 0:
            general_average_confidence_square = 0.0

        if general_average_confidence_linear > 1:
            general_average_confidence_linear = 1.0
        elif general_average_confidence_linear < 0:
            general_average_confidence_linear = 0.0

        # 元の手法：五つのラベルを0から1まで平均的に分ける
        # factuality_label_index = int(general_average_confidence // 0.2)

        # 修正手法：mostly-true/half-true/mostly-falseはpartly-trueの一部として扱ったら、False(0-0.33);True(0.67-1);
        # Partly-true(0.34 - 0.66)の方が正しいかもしれません。
        if general_average_confidence <= 0.33:
            factuality_label_index = 0
        elif general_average_confidence >= 0.67:
            factuality_label_index = 2
        else:
            factuality_label_index = 1
        factuality_label = factuality_labels[factuality_label_index]

        if general_average_confidence_square <= 0.33:
            factuality_label_index = 0
        elif general_average_confidence_square >= 0.67:
            factuality_label_index = 2
        else:
            factuality_label_index = 1
        factuality_label_square = factuality_labels[factuality_label_index]

        if general_average_confidence_linear <= 0.33:
            factuality_label_index = 0
        elif general_average_confidence_linear >= 0.67:
            factuality_label_index = 2
        else:
            factuality_label_index = 1
        factuality_label_linear = factuality_labels[factuality_label_index]

        os.makedirs(os.path.dirname(confidence_comparison_save_dir), exist_ok=True)
        with open(confidence_comparison_save_dir, mode="a", encoding="utf-8") as comparison_results:
            comparison_results.write(factuality_label + " " +
                                     factuality_label_square + " " +
                                     factuality_label_linear + "\n")

        # 元の投稿と全体的な確信度を合わせて、dictとして戻る
        post_and_confidence = {
            "post": original_post,
            "confidence": general_average_confidence,
            "factuality": factuality_label
        }

        self.status.update(status=200, message="successfully return original post and its confidence")
        return post_and_confidence

    def judge_original_post_factuality_binary(self, claims_and_sub_confidence, original_post: str):
        """
        最初の投稿の信憑性を計算するメソッドです。二項分類用です。

        ユーザの投稿からいくつかの主張が抽出され、そちらの確信度を別々で計算され、最後に合わせて元の投稿の信憑性を測る。
        計算で得られる信憑性は二項分類です。
        :param original_post: 元の投稿です。
        :param claims_and_sub_confidence: list[dict{"claim", "sub_confidence"}]
        claimは抽出された主張であり、sub_confidenceは正規化された主張の確信度である。0と近いのは誤り（あやまり）であり、1と近いのは事実である
        :return: post_and_confidence: dict{"post", "confidence", "factuality"}
        全体的な確信度は各主張の確信度の平均値です。
        factualityはこの投稿の確信度を示すラベルです。今は五つのラベルがあります："True", "mostly-true", "half-true",
        "barely-true", and "False"。
        """
        self.status.update(status=500, message="error in judge_original_claim_factuality")

        # for the ablation experiments
        confidence_comparison_save_dir = "./experiment_results/SciFact/confidence_comparison_binary_save_dir.txt"

        # claims_and_sub_confidenceからsub_confidenceだけを抽出する
        sub_confidence_list = [claim_and_sub_confidence["sub_confidence"] for
                               claim_and_sub_confidence in claims_and_sub_confidence]

        # for the ablation experiments
        sub_confidence_square_list = [claim_and_sub_confidence["sub_confidence_square"] for
                                      claim_and_sub_confidence in claims_and_sub_confidence]
        sub_confidence_linear_list = [claim_and_sub_confidence["sub_confidence_linear"] for
                                      claim_and_sub_confidence in claims_and_sub_confidence]

        # 元の投稿の全体的な確信度を計算する。0から1までの値です。
        confidence_sum = sum(sub_confidence_list)
        confidence_sum_upper = len(sub_confidence_list)
        confidence_sum_lower = 0
        general_average_confidence = self.confidence_normalization(confidence_real=confidence_sum,
                                                                   confidence_upper_limit=confidence_sum_upper,
                                                                   confidence_lower_limit=confidence_sum_lower)

        # for the ablation experiments
        confidence_sum_square = sum(sub_confidence_square_list)
        general_average_confidence_square = self.confidence_normalization(confidence_real=confidence_sum_square,
                                                                          confidence_upper_limit=confidence_sum_upper,
                                                                          confidence_lower_limit=confidence_sum_lower)

        confidence_sum_linear = sum(sub_confidence_linear_list)
        general_average_confidence_linear = self.confidence_normalization(confidence_real=confidence_sum_linear,
                                                                          confidence_upper_limit=confidence_sum_upper,
                                                                          confidence_lower_limit=confidence_sum_lower)
        if general_average_confidence > 0.5:
            factuality_label = "TRUE"
        else:
            factuality_label = "FALSE"

        if general_average_confidence_square > 0.5:
            factuality_label_square = "TRUE"
        else:
            factuality_label_square = "FALSE"

        if general_average_confidence_linear > 0.5:
            factuality_label_linear = "TRUE"
        else:
            factuality_label_linear = "FALSE"

        os.makedirs(os.path.dirname(confidence_comparison_save_dir), exist_ok=True)
        with open(confidence_comparison_save_dir, mode="a", encoding="utf-8") as comparison_results:
            comparison_results.write(factuality_label + " " +
                                     factuality_label_square + " " +
                                     factuality_label_linear + "\n")

        # 元の投稿と全体的な確信度を合わせて、dictとして戻る
        post_and_confidence = {
            "post": original_post,
            "confidence": general_average_confidence,
            "factuality": factuality_label
        }

        self.status.update(status=200, message="successfully return original post and its confidence")
        return post_and_confidence
