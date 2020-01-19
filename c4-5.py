import numpy as np 
import pandas as pd 

data = pd.read_csv('AUDIOLOGY.csv')




class Node:

    def __init__(self,label,unique_vals,IsLeaf):
        self.label = label
        self.unique_vals = unique_vals
        self.IsLeaf = IsLeaf
        self.children = []
        self.parent_node = None 
    



class C45:

    def __init__(self,df,target_col):

        self.data = df
        self.node_dict ={}
        self.target_col = target_col
        self.total_entropy = self.entropy_val(df[target_col],IsSeries=True)
        self.to_split_cols = list(df.columns)
        self.to_split_cols.remove(target_col)


    def entropy_val(self,target,IsSeries):
        if(IsSeries):
            attributes = list(target.value_counts().values)
            probabilities = attributes/sum(attributes)
            entropy=0
            for p in probabilities:
                entropy= entropy - p*np.log2(p)

        else:
            entropy =0
            for p in target:
                entropy = entropy - p*np.log2(p)

        return entropy 

    def split_info_val(self,df_list):
        prob_dataframe = [] 
        for df in df_list:
            prob_dataframe.append(df.shape[0]/self.data.shape[0])

        split_info = self.entropy_val(prob_dataframe,IsSeries=False)
        return split_info


    def gain_ratio_val(self,feature,df_):
        df_list = [] 
        unique_feature_vals = list(df_[feature].unique())

        for value in unique_feature_vals:
            temp_df = df_[df_[feature]==value]
            df_list.append(temp_df)

        entropy_split = 0
        for dataframe in df_list:
            prob_attribute = dataframe.shape[0]/df_.shape[0]
            entropy_of_dataframe = self.entropy_val(dataframe[self.target_col],IsSeries=True)
            entropy_split = entropy_split + prob_attribute*entropy_of_dataframe


        gain = self.entropy_val(df_[self.target_col],IsSeries=True) - entropy_split

        split_information = self.split_info_val(df_list)

        gain_ratio = gain/split_information

        return gain_ratio

    def best_split_val(self,df_):
        
        gain_ratios = []
        for feature in self.to_split_cols:
            gain_ratios.append(self.gain_ratio_val(feature,df_))

        max_index = gain_ratios.index(max(gain_ratios))
        best_split_column = self.to_split_cols[max_index]
        return best_split_column

    def recursive_build_tree(self,data_frame):

        if len(self.to_split_cols)==0:
            
            most_probable_value = max(set(list(data_frame[self.target_col])),key=list(data_frame[self.target_col]).count)
            leaf = Node('leaf',most_probable_value,IsLeaf=True)
            leaf.children = leaf.unique_vals
            return leaf

        if (self.entropy_val(data_frame[self.target_col],IsSeries=True)==0):
            
            leaf = Node('leaf',data_frame[self.target_col].unique(),IsLeaf=True)
            leaf.children = leaf.unique_vals
            return leaf

        else:
            best_split_attribute = self.best_split_val(data_frame)
            print(best_split_attribute)
            
            if best_split_attribute in self.to_split_cols:
                
                self.to_split_cols.remove(best_split_attribute)
                
                #print(self.to_split_cols)
            split_node = Node(label=best_split_attribute,unique_vals=data_frame[best_split_attribute].unique(),IsLeaf=False)
            
            
            for val in split_node.unique_vals:
                temp_df = data_frame[data_frame[best_split_attribute]==val]
                split_node.children.append(self.recursive_build_tree(temp_df))
                print('done')
            
            
            self.parent_node = split_node 
            return split_node 
        
    def predict(self,test_df):
        predictions_val = []
        for index, row in test_df.iterrows():
            target = self.parse_node(row,self.parent_node)
            predictions_val.append(target)
        
        
        
        predictions=[]
        for i in predictions_val:
            try:
                if (i).dtype == 'O':
                    predictions.append(list(i)[0])
            except:
                    predictions.append(i)

        return predictions

    def parse_node(self,row,node):
        if(node.IsLeaf):
            return node.unique_vals

        for index,feature in enumerate(node.unique_vals):
            if row[node.label]==feature:
                if self.parse_node(row,node.children[index])== None:
                    return 'NA'
                else:
                    return self.parse_node(row,node.children[index])
