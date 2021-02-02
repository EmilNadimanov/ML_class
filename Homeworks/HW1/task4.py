#!/usr/bin/env python3
from task1 import Tree

class MyDecisionTree(Tree):

    def leaf(data):
        '''Create a leaf tree'''
        return MyDecisionTree(data=data)


    def single_feature_score(data, goal, feature):
        truefalse = (data[goal]==data[feature])
        matches = truefalse.sum()
        total = truefalse.shape[0]
        return matches/total

    #shorter alias
    def sfs(data, goal, feature):
        return MyDecisionTree.single_feature_score(data, goal, feature)

    def decision_tree_train(data, features):
        score = dict()
        guess = data['ok'].mode()[0]
        if len(data['ok'].unique()) <= 1:
            return MyDecisionTree.leaf(guess)
        elif len(features)==1:
            return MyDecisionTree.leaf(guess)
        else:
            for f in features:
                NO = data.loc[data[f] == False]
                YES = data.loc[data[f] == True]
                score[f] = MyDecisionTree.sfs(NO, f, 'ok') + MyDecisionTree.sfs(YES, f, 'ok')

        top_f = max(score, key=score.get) # gets the key with the highest value in dict
        NO =  data.loc[data[top_f] == False]
        YES = data.loc[data[top_f] == True]
        left =  MyDecisionTree.decision_tree_train(NO,  set(features) ^ {top_f})
        right = MyDecisionTree.decision_tree_train(YES, set(features) ^ {top_f})
        return MyDecisionTree(data=top_f, left=left, right=right)

    # I assume that a test case is a row of a pandas dataframe
    def decision_tree_test(self, test_point):
        if self.is_leaf():
            return self.data
        else:
            if test_point[self.data] == False:
                return self.left.decision_tree_test(test_point)
            else:
                return self.right.decision_tree_test(test_point)

if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv('result')
    #I drop the unnecessary 'rating' column
    data_features_only = data.drop(columns=["rating"])
    my_tree = MyDecisionTree.decision_tree_train(data_features_only, set(data_features_only.columns)^{"ok"})
    print(my_tree)
    successes = 0

    for idx in range(data_features_only.shape[0]):
        result = my_tree.decision_tree_test(data_features_only.iloc[idx])
        label = data_features_only.iloc[idx]['ok']
        print(f"""FOR TEST CASE {idx}\n result is {result} \n label is {label}\n""")
        if result == label:
            successes += 1
    print("Decision tree result in a", successes / 20, "success rate")
    print("Comparing to sinle feature score:")
    for f in data_features_only.columns:
        print(f, MyDecisionTree.sfs(data, "ok", f))
