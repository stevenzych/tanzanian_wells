# Four basic functions used to help clean and sort data.
# Pandas-dependent. So sue me.

def many_count(ls, df, end=5):
    '''
    Returns value counts for n-given columns when
    provided a list.
    
    ---
    
    ls
        List of column names.
        
    df
        Dataframe.
    
    end
        How many most-frequent values to return. Defaults to 5.
    '''
    for i in ls:
        print(df[i].value_counts(normalize=True).iloc[:end])
        print('---------------')
    
    
    
def many_unique(ls, df):
    '''
    Returns all unique values for n-given columns when
    provided a list.
        
    ---
    
    ls
        List of column names.
        
    df
        Dataframe.
    '''
    for i in ls:
        print(df[i].unique())
        print('---------------')
    return



def freq_rep(ls, df):
    '''
    Replaces all values in n-columns with the total number of instances of
    that value within the column. This function is ALWAYS inplace, and will
    change your dataframe.
        
    ---
    
    ls
        List of column names.
        
    df
        Dataframe.
    '''
    for i in ls:
        vc = df[i].value_counts()
        names = list(vc.index)
        values = list(vc.values)
        df[i].replace(names, values, inplace=True)
        df.rename(columns = {i : f'{i}_freq'}, inplace=True)
    return



def top_dummies(ls, df, number=5, drop=True):
    '''
    Creates dummy variables for n-most-frequent values in k-columns. Also
    drops original column by default. Only makes dummies based on strings.
    At default, returns 5 dummy columns for 5 most frequent values in the
    given column(s) and drops the original(s).
        
    ---
    
    ls
        List of column names.
        
    df
        Dataframe.
        
    number
        Number of dummy columns to return. Defaults to 5.
        
    drop
        Whether or not to drop the original column from which the dummy
        variables were made. Defaults to True.
    '''
    for x in ls:
        for i in df[x].value_counts().index[:number]:
            df[i.replace(" ", "_")] = (df[x] == i).astype(int)
        if drop == True:
            df.drop(x, axis=1, inplace=True)
    return