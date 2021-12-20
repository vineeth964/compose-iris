import os
import pickle
from sklearn.naive_bayes import GaussianNB

# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

bug_classes = {0: False, 1: True}
bug_r_classes = {y: x for x, y in bug_classes.items()}

# function to process data and return it in correct format
def process_data(data):
    processed = [
        {
            "sepal_length": d.sepal_length,
            "sepal_width": d.sepal_length,
            "petal_length": d.petal_length,
            "petal_width": d.petal_width,
            "flower_class": d.flower_class,
        }
        for d in data
    ]

    return processed

def process_bug_data(data):
    processed = [
        {
            "lines_of_code": d.lines_of_code,
            "cyclomatic_complexity": d.cyclomatic_complexity,
            "essential_complexity": d.essential_complexity,
            "design_complexity": d.design_complexity,
            "totalo_perators_operands": d.totalo_perators_operands,
            "volume": d.volume,
            "program_length": d.program_length,
            "difficulty": d.difficulty,
            "intelligence" : d.intelligence,
            "effort" : d.effort,
            "b": d.b,
            "time_estimator": d.time_estimator,
            "lOCode": d.lOCode,
            "lOComment": d.lOComment,
            "lOBlank": d.lOBlank,
            "lOCodeAndComment": d.lOCodeAndComment,
            "uniq_Op": d.uniq_Op,
            "uniq_Opnd": d.uniq_Opnd,
            "total_Op": d.total_Op,
            "total_Opnd": d.total_Opnd,
            "branchCount": d.branchCount,
            "defects": d.defects,
        }
        for d in data
    ]

    return processed
