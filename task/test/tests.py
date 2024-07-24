import ast
from hstest.stage_test import List
from hstest import *

correct_answer = 0.7890909090909091

class OneHotTest(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):

        reply = reply.strip()

        if len(reply) == 0:
            return CheckResult.wrong("No output was printed")

        if len(reply.split('\n')) != 1:
            return CheckResult.wrong('The number of answers supplied does not equal 1')

        try:
            user_answer= ast.literal_eval(reply)
        except Exception as e:
            return CheckResult.wrong(f"Seems that output is in wrong format.\n"
                                     f"Make sure you use only the following Python structures in the output: string, int, float, list, dictionary")

        if not isinstance(user_answer, float):
            return CheckResult.wrong(f'Print answer as a float')

        if user_answer > correct_answer + 0.01 * correct_answer or user_answer < correct_answer - 0.01 * correct_answer:
            return CheckResult.wrong(f'Seems like your answer is not correct.')

        return CheckResult.correct()


if __name__ == '__main__':
    OneHotTest().run_tests()