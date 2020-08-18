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
        print(df[i].value_counts(normalize=True)[:end])
        print('---------------')
    return
    
    
    
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