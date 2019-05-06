#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 22:32:42 2019

@author: abraham
"""

#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl;
import matplotlib.pyplot as plt;
import numpy as np;
import gzip;
from io import StringIO
import math
#import io


# 1.Reading data

# In[7]:

def parse_header_of_csv(csv_str):
    # Isolate the headline columns:
    headline = csv_str[:csv_str.index('\n')];
    columns = headline.split(',');

    # The first column should be timestamp:
    assert columns[0] == 'timestamp';
    # The last column should be label_source:
    assert columns[-1] == 'label_source';
    
    # Search for the column of the first label:
    for (ci,col) in enumerate(columns):
        if col.startswith('label:'):
            first_label_ind = ci;
            break;
        pass;

    # Feature columns come after timestamp and before the labels:
    feature_names = columns[1:first_label_ind];
    # Then come the labels, till the one-before-last column:
    label_names = columns[first_label_ind:-1];
    for (li,label) in enumerate(label_names):
        # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
        assert label.startswith('label:');
        label_names[li] = label.replace('label:','');
        pass;
    
    return (feature_names,label_names);

def parse_body_of_csv(csv_str,n_features):
    # Read the entire CSV body into a single numeric matrix:
    
    full_table = np.loadtxt(StringIO(csv_str),delimiter=',',skiprows=1);
    
    # Timestamp is the primary key for the records (examples):
    timestamps = full_table[:,0].astype(int);
    
    # Read the sensor features:
    X = full_table[:,1:(n_features+1)];
    
    # Read the binary label values, and the 'missing label' indicators:
    trinary_labels_mat = full_table[:,(n_features+1):-1]; # This should have values of either 0., 1. or NaN
    M = np.isnan(trinary_labels_mat); # M is the missing label matrix
    Y = np.where(M,0,trinary_labels_mat) > 0.; # Y is the label matrix
    
    return (X,Y,M,timestamps);

def read_user_data(uuid):
    user_data_file = '%s.features_labels.csv.gz' % uuid;

    # Read the entire csv file of the user:
    with gzip.open(user_data_file,'rb') as fid:
        csv_str = fid.read();
        
    csv_str = csv_str.decode('ASCII')
    feature_names, label_names = parse_header_of_csv(csv_str);
    n_features = len(feature_names);
    (X,Y,M,timestamps) = parse_body_of_csv(csv_str,n_features);

    return (X,Y,M,timestamps,feature_names,label_names);

# In[8]:
UUID_length=  2
cont_num=4
input_init=1
F1_score_phone=np.zeros(shape=(cont_num))
balanced_accuracy_phone= np.zeros(shape=(cont_num,UUID_length))
accuracy_phone=np.zeros(shape=(cont_num,UUID_length))
    
uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/FDAA70A1-42A3-4E3F-9AE3-3FDA412E03BF';
(X,Y,M,timestamps,feature_names,label_names) = read_user_data(uuid);


# In[4]:


print (label_names)


n_examples_per_label = np.sum(Y,axis=0);
labels_and_counts = zip(label_names,n_examples_per_label);
sorted_labels_and_counts = sorted(labels_and_counts,reverse=True,key=lambda pair:pair[1]);
for (label,count) in sorted_labels_and_counts:
    print ("label %s - %d minutes" % (label,count));
    pass;
    
    
    # In[6]:
    

def get_label_pretty_name(label):
    if label == 'FIX_walking':
        return 'Walking';
    if label == 'FIX_running':
        return 'Running';
    if label == 'LOC_main_workplace':
        return 'At main workplace';
    if label == 'OR_indoors':
        return 'Indoors';
    if label == 'OR_outside':
        return 'Outside';
    if label == 'LOC_home':
        return 'At home';
    if label == 'FIX_restaurant':
        return 'At a restaurant';
    if label == 'OR_exercise':
        return 'Exercise';
    if label == 'LOC_beach':
        return 'At the beach';
    if label == 'OR_standing':
        return 'Standing';
    if label == 'WATCHING_TV':
        return 'Watching TV'
    
    if label.endswith('_'):
        label = label[:-1] + ')';
        pass;
    
    label = label.replace('__',' (').replace('_',' ');
    label = label[0] + label[1:].lower();
    label = label.replace('i m','I\'m');
    return label;

    
    # In[7]:


print ("How many examples does this user have for each contex-label:");
print ("-"*20);
for (label,count) in sorted_labels_and_counts:
    print ("%s - %d minutes" % (get_label_pretty_name(label),count));
    pass;


# In[8]:


def figure__pie_chart(Y,label_names,labels_to_display,title_str,ax):
    portion_of_time = np.mean(Y,axis=0);
    portions_to_display = [portion_of_time[label_names.index(label)] for label in labels_to_display];
    pretty_labels_to_display = [get_label_pretty_name(label) for label in labels_to_display];
    ax.pie(portions_to_display,labels=pretty_labels_to_display,autopct='%.2f%%');
    ax.axis('equal');
    plt.title(title_str);
    return;


# In[9]:


    print("Since the collection of labels relied on self-reporting in-the-wild, the labeling may be incomplete.");
    print("For instance, the users did not always report the position of the phone.");


# In[10]:


def get_actual_date_labels(tick_seconds):
    import datetime;
    import pytz;

    time_zone = pytz.timezone('US/Pacific'); # Assuming the data comes from PST time zone
    weekday_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
    datetime_labels = [];
    for timestamp in tick_seconds:
        tick_datetime = datetime.datetime.fromtimestamp(timestamp,tz=time_zone);
        weekday_str = weekday_names[tick_datetime.weekday()];
        time_of_day = tick_datetime.strftime('%I:%M%p');
        datetime_labels.append('%s\n%s' % (weekday_str,time_of_day));
        pass;

    return datetime_labels;
    
def figure__context_over_participation_time(timestamps,Y,label_names,labels_to_display,label_colors,use_actual_dates=False):

    fig = plt.figure(figsize=(10,7),facecolor='white');
    ax = plt.subplot(1,1,1);
    
    seconds_in_day = (60*60*24);

    ylabels = [];
    ax.plot(timestamps,len(ylabels)*np.ones(len(timestamps)),'|',color='0.5',label='(Collected data)');
    ylabels.append('(Collected data)');

    for (li,label) in enumerate(labels_to_display):
        lind = label_names.index(label);
        is_label_on = Y[:,lind];
        label_times = timestamps[is_label_on];

        label_str = get_label_pretty_name(label);
        ax.plot(label_times,len(ylabels)*np.ones(len(label_times)),'|',color=label_colors[li],label=label_str);
        ylabels.append(label_str);
        pass;

    tick_seconds = range(timestamps[0],timestamps[-1],seconds_in_day);
    if use_actual_dates:
        tick_labels = get_actual_date_labels(tick_seconds);
        plt.xlabel('Time in San Diego',fontsize=14);
        pass;
    else:
        tick_labels = (np.array(tick_seconds - timestamps[0]).astype(float) / float(seconds_in_day)).astype(int);
        plt.xlabel('days of participation',fontsize=14);
        pass;
    
    ax.set_xticks(tick_seconds);
    ax.set_xticklabels(tick_labels,fontsize=14);

    ax.set_yticks(range(len(ylabels)));
    ax.set_yticklabels(ylabels,fontsize=14);

    ax.set_ylim([-1,len(ylabels)]);
    ax.set_xlim([min(timestamps),max(timestamps)]);
    
    return;


# In[11]:


print("Here is a track of when the user was engaged in different contexts.");
print("The bottom row (gray) states when sensors were recorded (the data-collection app was not running all the time).");
print("The context-labels annotations were self-reported by ther user (and then cleaned by the researchers).")

labels_to_display = ['LYING_DOWN','LOC_home','LOC_main_workplace','SITTING','OR_standing','FIX_walking',                     'IN_A_CAR','ON_A_BUS','EATING'];
label_colors = ['g','y','b','c','m','b','r','k','purple'];
#figure__context_over_participation_time(timestamps,Y,label_names,labels_to_display,label_colors);

# In[13]:


def jaccard_similarity_for_label_pairs(Y):
    (n_examples,n_labels) = Y.shape;
    Y = Y.astype(int);
    # For each label pair, count cases of:
    # Intersection (co-occurrences) - cases when both labels apply:
    both_labels_counts = np.dot(Y.T,Y);
    # Cases where neither of the two labels applies:
    neither_label_counts = np.dot((1-Y).T,(1-Y));
    # Union - cases where either of the two labels (or both) applies (this is complement of the 'neither' cases):
    either_label_counts = n_examples - neither_label_counts;
    # Jaccard similarity - intersection over union:
    J = np.where(either_label_counts > 0, both_labels_counts.astype(float) / either_label_counts, 0.);
    return J;


# In[14]:


J = jaccard_similarity_for_label_pairs(Y);

print("Label-pairs with higher color value tend to occur together more.");


# In[15]:


def get_sensor_names_from_features(feature_names):
    feat_sensor_names = np.array([None for feat in feature_names]);
    for (fi,feat) in enumerate(feature_names):
        if feat.startswith('raw_acc'):
            feat_sensor_names[fi] = 'Acc';
            pass;
        elif feat.startswith('proc_gyro'):
            feat_sensor_names[fi] = 'Gyro';
            pass;
        elif feat.startswith('raw_magnet'):
            feat_sensor_names[fi] = 'Magnet';
            pass;
        elif feat.startswith('watch_acceleration'):
            feat_sensor_names[fi] = 'WAcc';
            pass;
        elif feat.startswith('watch_heading'):
            feat_sensor_names[fi] = 'Compass';
            pass;
        elif feat.startswith('location'):
            feat_sensor_names[fi] = 'Loc';
            pass;
        elif feat.startswith('location_quick_features'):
            feat_sensor_names[fi] = 'Loc';
            pass;
        elif feat.startswith('audio_naive'):
            feat_sensor_names[fi] = 'Aud';
            pass;
        elif feat.startswith('audio_properties'):
            feat_sensor_names[fi] = 'AP';
            pass;
        elif feat.startswith('discrete'):
            feat_sensor_names[fi] = 'PS';
            pass;
        elif feat.startswith('lf_measurements'):
            feat_sensor_names[fi] = 'LF';
            pass;
        else:
            raise ValueError("!!! Unsupported feature name: %s" % feat);

        pass;

    return feat_sensor_names;    


# In[16]:


feat_sensor_names = get_sensor_names_from_features(feature_names);

for (fi,feature) in enumerate(feature_names):
    print("%3d) %s %s" % (fi,feat_sensor_names[fi].ljust(10),feature));
    pass;


# In[17]:


def figure__feature_track_and_hist(X,feature_names,timestamps,feature_inds):
    seconds_in_day = (60*60*24);
    days_since_participation = (timestamps - timestamps[0]) / float(seconds_in_day);
    
    for ind in feature_inds:
        feature = feature_names[ind];
        feat_values = X[:,ind];
        
        fig = plt.figure(figsize=(10,3),facecolor='white');
        
        ax1 = plt.subplot(1,2,1);
        ax1.plot(days_since_participation,feat_values,'.-',markersize=3,linewidth=0.1);
        plt.xlabel('days of participation');
        plt.ylabel('feature value');
        plt.title('%d) %s\nfunction of time' % (ind,feature));
        
        ax1 = plt.subplot(1,2,2);
        existing_feature = np.logical_not(np.isnan(feat_values));
        ax1.hist(feat_values[existing_feature],bins=30);
        plt.xlabel('feature value');
        plt.ylabel('count');
        plt.title('%d) %s\nhistogram' % (ind,feature));
        
        pass;
    
    return;


# In[18]:


feature_inds = [0,102,133,148,157,158];
#figure__feature_track_and_hist(X,feature_names,timestamps,feature_inds);


# In[19]:


print("The phone-state (PS) features are represented as binary indicators:");
feature_inds = [205,223];
#figure__feature_track_and_hist(X,feature_names,timestamps,feature_inds);


# Relation between sensor fetures and context label

# In[20]:


def figure__feature_scatter_for_labels(X,Y,feature_names,label_names,feature1,feature2,label2color_map):
    feat_ind1 = feature_names.index(feature1);
    feat_ind2 = feature_names.index(feature2);
    example_has_feature1 = np.logical_not(np.isnan(X[:,feat_ind1]));
    example_has_feature2 = np.logical_not(np.isnan(X[:,feat_ind2]));
    example_has_features12 = np.logical_and(example_has_feature1,example_has_feature2);
    
    fig = plt.figure(figsize=(12,5),facecolor='white');
    ax1 = plt.subplot(1,2,1);
    ax2 = plt.subplot(2,2,2);
    ax3 = plt.subplot(2,2,4);
    
    for label in label2color_map.keys():
        label_ind = label_names.index(label);
        pretty_name = get_label_pretty_name(label);
        color = label2color_map[label];
        style = '.%s' % color;
        
        is_relevant_example = np.logical_and(example_has_features12,Y[:,label_ind]);
        count = sum(is_relevant_example);
        feat1_vals = X[is_relevant_example,feat_ind1];
        feat2_vals = X[is_relevant_example,feat_ind2];
        ax1.plot(feat1_vals,feat2_vals,style,markersize=5,label=pretty_name);
        
        ax2.hist(X[is_relevant_example,feat_ind1],bins=20,normed=True,color=color,alpha=0.5,label='%s (%d)' % (pretty_name,count));
        ax3.hist(X[is_relevant_example,feat_ind2],bins=20,normed=True,color=color,alpha=0.5,label='%s (%d)' % (pretty_name,count));
        pass;
    
    ax1.set_xlabel(feature1);
    ax1.set_ylabel(feature2);
    
    ax2.set_title(feature1);
    ax3.set_title(feature2);
    
    ax2.legend(loc='best');
    
    return;


# In[21]:


feature1 = 'proc_gyro:magnitude_stats:time_entropy';#raw_acc:magnitude_autocorrelation:period';
feature2 = 'raw_acc:3d:mean_y';
label2color_map = {'PHONE_IN_HAND':'b','PHONE_ON_TABLE':'g'};
#figure__feature_scatter_for_labels(X,Y,feature_names,label_names,feature1,feature2,label2color_map);


# In[22]:


feature1 = 'watch_acceleration:magnitude_spectrum:log_energy_band1';
feature2 = 'watch_acceleration:3d:mean_z';
label2color_map = {'FIX_walking':'b','WATCHING_TV':'g'};
#figure__feature_scatter_for_labels(X,Y,feature_names,label_names,feature1,feature2,label2color_map);


# 2.Design models and apply them in a single user dataset

# In[23]:


import sklearn.linear_model;

def project_features_to_selected_sensors(X,feat_sensor_names,sensors_to_use):
    use_feature = np.zeros(len(feat_sensor_names),dtype=bool);
    for sensor in sensors_to_use:
        is_from_sensor = (feat_sensor_names == sensor);
        use_feature = np.logical_or(use_feature,is_from_sensor);
        pass;
    X = X[:,use_feature];
    return X;

def estimate_standardization_params(X_train):
    mean_vec = np.nanmean(X_train,axis=0);
    std_vec = np.nanstd(X_train,axis=0);
    return (mean_vec,std_vec);

def standardize_features(X,mean_vec,std_vec):
    # Subtract the mean, to centralize all features around zero:
    X_centralized = X - mean_vec.reshape((1,-1));
    # Divide by the standard deviation, to get unit-variance for all features:
    # * Avoid dividing by zero, in case some feature had estimate of zero variance
    normalizers = np.where(std_vec > 0., std_vec, 1.).reshape((1,-1));
    X_standard = X_centralized / normalizers;
    return X_standard;
#%%
    
def validate_column_names_are_consistent(old_column_names,new_column_names):
    if len(old_column_names) != len(new_column_names):
        raise ValueError("!!! Inconsistent number of columns.");
        
    for ci in range(len(old_column_names)):
        if old_column_names[ci] != new_column_names[ci]:
            raise ValueError("!!! Inconsistent column %d) %s != %s" % (ci,old_column_names[ci],new_column_names[ci]));
        pass;
    return;
#%%

#train part of svm
import sklearn.svm

def train_svm_model(X_train,Y_train,M_train,feat_sensor_names,label_names,sensors_to_use,target_label):
    # Project the feature matrix to the features from the desired sensors:
    X_train = project_features_to_selected_sensors(X_train,feat_sensor_names,sensors_to_use);
    print("== Projected the features to %d features from the sensors: %s" % (X_train.shape[1],', '.join(sensors_to_use)));

    # It is recommended to standardize the features (subtract mean and divide by standard deviation),
    # so that all their values will be roughly in the same range:
    (mean_vec,std_vec) = estimate_standardization_params(X_train);
    X_train = standardize_features(X_train,mean_vec,std_vec);
    print (X_train.shape)
    # The single target label:
    label_ind = label_names.index(target_label);
    y = Y_train[:,label_ind];
    missing_label = M_train[:,label_ind];
    existing_label = np.logical_not(missing_label);
    
    # Select only the examples that are not missing the target label:
    X_train = X_train[existing_label,:];
    y = y[existing_label];
    
    print (X_train.shape)
    print (y.shape)
    
    #If there is no example
    if len(y) == 0:
        svm_model = None;
        # Assemble all the parts of the model:
        model = {            'sensors_to_use':sensors_to_use,            'target_label':target_label,            'mean_vec':mean_vec,            'std_vec':std_vec,            'svm_model':svm_model};
        return model;
    
    # Also, there may be missing sensor-features (represented in the data as NaN).
    # You can handle those by imputing a value of zero (since we standardized, this is equivalent to assuming average value).
    # You can also further select examples - only those that have values for all the features.
    # For this tutorial, let's use the simple heuristic of zero-imputation:
    X_train[np.isnan(X_train)] = 0.;
    #y = y[~np.isnan(X_train).any(axis=1)]
    #X_train= X_train[~np.isnan(X_train).any(axis=1)]
    
    print("== Training with %d examples. For label '%s' we have %d positive and %d negative examples." %           (len(y),get_label_pretty_name(target_label),sum(y),sum(np.logical_not(y))) );
    
    # Now, we have the input features and the ground truth for the output label.
    # We can train a logistic regression model.
    
    # Typically, the data is highly imbalanced, with many more negative examples;
    # To avoid a trivial classifier (one that always declares 'no'), it is important to counter-balance the pos/neg classes:
    svm_model = sklearn.svm.SVC(kernel='rbf',tol=0.0001,class_weight='balanced');
    svm_model.probability = True
    svm_model.fit(X_train,y);
  
    # Assemble all the parts of the model:
    model = {            'sensors_to_use':sensors_to_use,            'target_label':target_label,            'mean_vec':mean_vec,            'std_vec':std_vec,            'svm_model':svm_model};
    
    return model;


# In[38]:





# In[39]:


#test part of svm
def test_svm_model(X_test,Y_test,M_test,timestamps,feat_sensor_names,label_names,model):
    #Check if the model is null:
    if model['svm_model'] == None:
        return (1.0,1.0,1.0);
    
    # Project the feature matrix to the features from the sensors that the classifier is based on:
    X_test = project_features_to_selected_sensors(X_test,feat_sensor_names,model['sensors_to_use']);
    print("== Projected the features to %d features from the sensors: %s" % (X_test.shape[1],', '.join(model['sensors_to_use'])));

    # We should standardize the features the same way the train data was standardized:
    X_test = standardize_features(X_test,model['mean_vec'],model['std_vec']);
    
    # The single target label:
    label_ind = label_names.index(model['target_label']);
    y = Y_test[:,label_ind];
    missing_label = M_test[:,label_ind];
    existing_label = np.logical_not(missing_label);
    
    # Select only the examples that are not missing the target label:
    X_test = X_test[existing_label,:];
    y = y[existing_label];
    timestamps = timestamps[existing_label];
    
    print (X_test.shape);
    print (y.shape);
    
    #If there is no example
    if len(y) == 0:
        return (1.0,1.0,1.0);
    
    # Do the same treatment for missing features as done to the training data:
    X_test[np.isnan(X_test)] = 0.;
    #y = y[~np.isnan(X_test).any(axis=1)]
    #X_test= X_test[~np.isnan(X_test).any(axis=1)]
    
    print("== Testing with %d examples. For label '%s' we have %d positive and %d negative examples." %           (len(y),get_label_pretty_name(target_label),sum(y),sum(np.logical_not(y))) );
    
    # Preform the prediction:
    y_pred = model['svm_model'].predict(X_test);
    y_pred_prob = model['svm_model'].predict_proba(X_test);
   
    
    # Naive accuracy (correct classification rate):
    accuracy = np.mean(y_pred == y);
    
    # Count occorrences of true-positive, true-negative, false-positive, and false-negative:
    tp = np.sum(np.logical_and(y_pred,y));
    tn = np.sum(np.logical_and(np.logical_not(y_pred),np.logical_not(y)));
    fp = np.sum(np.logical_and(y_pred,np.logical_not(y)));
    fn = np.sum(np.logical_and(np.logical_not(y_pred),y));
    
    sensitivity = 0;
    specificity = 0;
    precision = 0;
    
    # Sensitivity (=recall=true positive rate) and Specificity (=true negative rate):
    if tp+fn != 0:
        sensitivity = float(tp) / (tp+fn);
    if tn+fp != 0:
        specificity = float(tn) / (tn+fp);
    
    # Balanced accuracy is a more fair replacement for the naive accuracy:
    balanced_accuracy = (sensitivity + specificity) / 2.;
    
    # Precision:
    # Beware from this metric, since it may be too sensitive to rare labels.
    # In the ExtraSensory Dataset, there is large skew among the positive and negative classes,
    # and for each label the pos/neg ratio is different.
    # This can cause undesirable and misleading results when averaging precision across different labels.
    if tp+fp != 0:
        precision = float(tp) / (tp+fp);
    
    print("-"*10);
    print('Accuracy*:         %.2f' % accuracy);
    print('Sensitivity (TPR): %.2f' % sensitivity);
    print('Specificity (TNR): %.2f' % specificity);
    print('Balanced accuracy: %.2f' % balanced_accuracy);
    print('Precision**:       %.2f' % precision);
    print("-"*10);
   #print('The probability is ' , y_pred_prob)
    
    return(accuracy,balanced_accuracy,precision,y_pred_prob,y);

#%%
    


#train part of random forest
import sklearn.ensemble
import numpy
from sklearn.utils import resample
import pandas as pd

def train_rf_model(X_train,Y_train,M_train,feat_sensor_names,label_names,sensors_to_use,target_label):
    # Project the feature matrix to the features from the desired sensors:
    X_train = project_features_to_selected_sensors(X_train,feat_sensor_names,sensors_to_use);
    print("== Projected the features to %d features from the sensors: %s" % (X_train.shape[1],', '.join(sensors_to_use)));

    # It is recommended to standardize the features (subtract mean and divide by standard deviation),
    # so that all their values will be roughly in the same range:
   # X_train = X_train[~np.isnan(X_train)]
    #X_train= X_train[np.logical_not(np.isnan(X_train))]
      #X_train_new= X_train[~np.isnan(X_train).any(axis=1)]
    (mean_vec,std_vec) = estimate_standardization_params(X_train);
    X_train = standardize_features(X_train,mean_vec,std_vec);
    print (X_train.shape)
    # The single target label:
    label_ind = label_names.index(target_label);
    y = Y_train[:,label_ind];
    missing_label = M_train[:,label_ind];
    existing_label = np.logical_not(missing_label);
    
    # Select only the examples that are not missing the target label:
    X_train = X_train[existing_label,:];
    y = y[existing_label];
    
    print (X_train.shape)
    print (y.shape)
    
    #If there is no example
    if len(y) == 0:
        rf_model = None;
        # Assemble all the parts of the model:
        model = { 'sensors_to_use':sensors_to_use,  'target_label':target_label,  'mean_vec':mean_vec,     'std_vec':std_vec,   'rf_model':rf_model};
        return model;
    
    # Also, there may be missing sensor-features (represented in the data as NaN).
    # You can handle those by imputing a value of zero (since we standardized, this is equivalent to assuming average value).
    # You can also further select examples - only those that have values for all the features.
    # For this tutorial, let's use the simple heuristic of zero-imputation:
   # X_train[np.isnan(X_train)] = 0.;
   # X_train.dropna(inplace=True)
    #X_train[np.isnan(X_train)] = 0.;
    y = y[~np.isnan(X_train).any(axis=1)]
    X_train= X_train[~np.isnan(X_train).any(axis=1)]
    
    #X_train_new = X_train[~np.isnan(X_train)]
    
    #X_train_new=X_train(~np.isnan(X_train))
    print("== Training with %d examples. For label '%s' we have %d positive and %d negative examples." %  (len(y),get_label_pretty_name(target_label),sum(y),sum(np.logical_not(y))) );
    
    # Now, we have the input features and the ground truth for the output label.
    # We can train a logistic regression model.
    
    # Typically, the data is highly imbalanced, with many more negative examples;
    # To avoid a trivial classifier (one that always declares 'no'), it is important to counter-balance the pos/neg classes:
   # Indicies of each class' observations
    #if sum(y) < sum(np.logical_not(y)):
    i_class0 = np.where(y == 1)[0]
    i_class1 = np.where(y == 0)[0]
    #else:
     #   i_class0 = np.where(y == 0)[0]
      #  i_class1 = np.where(y == 1)[0]
# Number of observations in each class
    n_class0 = len(i_class0)
    n_class1 = len(i_class1)

# For every observation in class 1, randomly sample from class 0 with replacement
    i_class0_upsampled = np.random.choice(i_class0, size=n_class1, replace=True)

# Join together class 0's upsampled target vector with class 1's target vector
    y=np.concatenate((y[i_class0_upsampled], y[i_class1]))
    X_train=np.concatenate((X_train[i_class0_upsampled], X_train[i_class1]))
    
    rf_model = sklearn.ensemble.RandomForestClassifier(n_estimators=10,class_weight='balanced');
   
    rf_model.fit(X_train,y);
    #rf_model.fit(X_train_new,y_new);
    
    # Assemble all the parts of the model:
    model = { 'sensors_to_use':sensors_to_use,  'target_label':target_label,    'mean_vec':mean_vec,    'std_vec':std_vec,      'rf_model':rf_model};
    
    return model;


# In[50]:


# In[51]:


#test part of random forest
def test_rf_model(X_test,Y_test,M_test,timestamps,feat_sensor_names,label_names,model):
    #Check if the model is null:
    if model['rf_model'] == None:
        return (1,1,1,1,1);
        #return (0.0,0.0,0.0,0.0,0.0);
    # Project the feature matrix to the features from the sensors that the classifier is based on:
    X_test = project_features_to_selected_sensors(X_test,feat_sensor_names,model['sensors_to_use']);
    print("== Projected the features to %d features from the sensors: %s" % (X_test.shape[1],', '.join(model['sensors_to_use'])));

    # We should standardize the features the same way the train data was standardized:
    X_test = standardize_features(X_test,model['mean_vec'],model['std_vec']);
    
    # The single target label:
    label_ind = label_names.index(model['target_label']);
    y = Y_test[:,label_ind];
    missing_label = M_test[:,label_ind];
    existing_label = np.logical_not(missing_label);
    
    # Select only the examples that are not missing the target label:
    X_test = X_test[existing_label,:];
    y = y[existing_label];
    timestamps = timestamps[existing_label];
    
    print (X_test.shape);
    print (y.shape);
    
    #If there is no example
    if len(y) == 0:
        return (1.0,1.0,1.0,1.0,1.0);
        #return (0.0,0.0,0.0,0.0,0.0);
    # Do the same treatment for missing features as done to the training data:
    y = y[~np.isnan(X_test).any(axis=1)]
    X_test= X_test[~np.isnan(X_test).any(axis=1)]
    #X_test[np.isnan(X_test)] = 0.0;
    #y_new=y;
    #X_test_new= X_test[~np.isnan(X_test).any(axis=1)]
    #y_new = y[~np.isnan(X_test).any(axis=1)]
    print("== Testing with %d examples. For label '%s' we have %d positive and %d negative examples." %           (len(y),get_label_pretty_name(target_label),sum(y),sum(np.logical_not(y))) );
    
    # Preform the prediction:
    y_pred = model['rf_model'].predict(X_test);
    y_pred_prob = model['rf_model'].predict_proba(X_test);
    #if sum(y_pred_prob[:,0]) == len(y_pred_prob):
     #   y_pred_prob=np.concatenate((y_pred_prob,np.zeros(shape=(len(y_pred_prob),1))),1)
    #y_pred = model['rf_model'].predict(X_test_new);
    #y_pred_prob = model['rf_model'].predict_proba(X_test_new);
    # Naive accuracy (correct classification rate):
    accuracy = np.mean(y_pred == y);
    
    # Count occorrences of true-positive, true-negative, false-positive, and false-negative:
    tp = np.sum(np.logical_and(y_pred,y));
    tn = np.sum(np.logical_and(np.logical_not(y_pred),np.logical_not(y)));
    fp = np.sum(np.logical_and(y_pred,np.logical_not(y)));
    fn = np.sum(np.logical_and(np.logical_not(y_pred),y));
    
    sensitivity = 0;
    specificity = 0;
    precision = 0;
    F1_score_value=0;
    
    # Sensitivity (=recall=true positive rate) and Specificity (=true negative rate):
    if tp+fn != 0:
        sensitivity = float(tp) / (tp+fn);
    if tn+fp != 0:
        specificity = float(tn) / (tn+fp);
    
    # Balanced accuracy is a more fair replacement for the naive accuracy:
    balanced_accuracy = (sensitivity + specificity) / 2.;
    
    # Precision:
    # Beware from this metric, since it may be too sensitive to rare labels.
    # In the ExtraSensory Dataset, there is large skew among the positive and negative classes,
    # and for each label the pos/neg ratio is different.
    # This can cause undesirable and misleading results when averaging precision across different labels.
    if tp+fp != 0:
        precision = float(tp) / (tp+fp);
    
    
    # Balanced accuracy is a more fair replacement for the naive accuracy:
    if (precision !=0) & (sensitivity!=0) :
        #F1_score_phone[i] [input_value]= 2 * (sensitivity * precision) / (sensitivity + precision)
        F1_score_value = 2 * (sensitivity * precision) / (sensitivity + precision)

    
    print("-"*10);
    print('Accuracy*:         %.2f' % accuracy);
    print('Sensitivity (TPR): %.2f' % sensitivity);
    print('Specificity (TNR): %.2f' % specificity);
    print('Balanced accuracy: %.2f' % balanced_accuracy);
    print('Precision**:       %.2f' % precision);
    print("-"*10);
    print('F1 score**:       %.2f' % F1_score_value);
    return(accuracy,balanced_accuracy,precision, y_pred_prob,  y);
    
#%%#train part of Extra forest
import sklearn.ensemble
import numpy

def train_EF_model(X_train,Y_train,M_train,feat_sensor_names,label_names,sensors_to_use,target_label):
    # Project the feature matrix to the features from the desired sensors:
    X_train = project_features_to_selected_sensors(X_train,feat_sensor_names,sensors_to_use);
    print("== Projected the features to %d features from the sensors: %s" % (X_train.shape[1],', '.join(sensors_to_use)));

    # It is recommended to standardize the features (subtract mean and divide by standard deviation),
    # so that all their values will be roughly in the same range:
   # X_train = X_train[~np.isnan(X_train)]
    #X_train= X_train[np.logical_not(np.isnan(X_train))]
      #X_train_new= X_train[~np.isnan(X_train).any(axis=1)]
    (mean_vec,std_vec) = estimate_standardization_params(X_train);
    X_train = standardize_features(X_train,mean_vec,std_vec);
    print (X_train.shape)
    # The single target label:
    label_ind = label_names.index(target_label);
    y = Y_train[:,label_ind];
    missing_label = M_train[:,label_ind];
    existing_label = np.logical_not(missing_label);
    
    # Select only the examples that are not missing the target label:
    X_train = X_train[existing_label,:];
    y = y[existing_label];
    
    print (X_train.shape)
    print (y.shape)
    
    #If there is no example
    if len(y) == 0:
        EF_model = None;
        # Assemble all the parts of the model:
        model = { 'sensors_to_use':sensors_to_use,  'target_label':target_label,  'mean_vec':mean_vec,     'std_vec':std_vec,   'EF_model':EF_model};
        return model;
    
    # Also, there may be missing sensor-features (represented in the data as NaN).
    # You can handle those by imputing a value of zero (since we standardized, this is equivalent to assuming average value).
    # You can also further select examples - only those that have values for all the features.
    # For this tutorial, let's use the simple heuristic of zero-imputation:
   # X_train[np.isnan(X_train)] = 0.;
   # X_train.dropna(inplace=True)
    #X_train[np.isnan(X_train)] = 0.;
    y = y[~np.isnan(X_train).any(axis=1)]
    X_train= X_train[~np.isnan(X_train).any(axis=1)]
    
    #X_train_new = X_train[~np.isnan(X_train)]
    
    #X_train_new=X_train(~np.isnan(X_train))
    print("== Training with %d examples. For label '%s' we have %d positive and %d negative examples." %  (len(y),get_label_pretty_name(target_label),sum(y),sum(np.logical_not(y))) );
    
    # Now, we have the input features and the ground truth for the output label.
    # We can train a logistic regression model.
    
    # Typically, the data is highly imbalanced, with many more negative examples;
    # To avoid a trivial classifier (one that always declares 'no'), it is important to counter-balance the pos/neg classes:
    EF_model = sklearn.ensemble.ExtraTreesClassifier(n_estimators=10,class_weight='balanced');
    EF_model.fit(X_train,y);
    #rf_model.fit(X_train_new,y_new);
    
    # Assemble all the parts of the model:
    model = { 'sensors_to_use':sensors_to_use,  'target_label':target_label,    'mean_vec':mean_vec,    'std_vec':std_vec,      'EF_model':EF_model};
    
    return model;


# In[50]:


# In[51]:


#test part of Extra forest
def test_EF_model(X_test,Y_test,M_test,timestamps,feat_sensor_names,label_names,model):
    #Check if the model is null:
    if model['EF_model'] == None:
        return (1,1,1,1,1);
        #return (0.0,0.0,0.0,0.0,0.0);
    # Project the feature matrix to the features from the sensors that the classifier is based on:
    X_test = project_features_to_selected_sensors(X_test,feat_sensor_names,model['sensors_to_use']);
    print("== Projected the features to %d features from the sensors: %s" % (X_test.shape[1],', '.join(model['sensors_to_use'])));

    # We should standardize the features the same way the train data was standardized:
    X_test = standardize_features(X_test,model['mean_vec'],model['std_vec']);
    
    # The single target label:
    label_ind = label_names.index(model['target_label']);
    y = Y_test[:,label_ind];
    missing_label = M_test[:,label_ind];
    existing_label = np.logical_not(missing_label);
    
    # Select only the examples that are not missing the target label:
    X_test = X_test[existing_label,:];
    y = y[existing_label];
    timestamps = timestamps[existing_label];
    
    print (X_test.shape);
    print (y.shape);
    
    #If there is no example
    if len(y) == 0:
        return (1.0,1.0,1.0,1.0,1.0);
        #return (0.0,0.0,0.0,0.0,0.0);
    # Do the same treatment for missing features as done to the training data:
    y = y[~np.isnan(X_test).any(axis=1)]
    X_test= X_test[~np.isnan(X_test).any(axis=1)]
    #X_test[np.isnan(X_test)] = 0.0;
    #y_new=y;
    #X_test_new= X_test[~np.isnan(X_test).any(axis=1)]
    #y_new = y[~np.isnan(X_test).any(axis=1)]
    print("== Testing with %d examples. For label '%s' we have %d positive and %d negative examples." %           (len(y),get_label_pretty_name(target_label),sum(y),sum(np.logical_not(y))) );
    
    # Preform the prediction:
    y_pred = model['EF_model'].predict(X_test);
    y_pred_prob = model['EF_model'].predict_proba(X_test);
    #if sum(y_pred_prob[:,0]) == len(y_pred_prob):
     #   y_pred_prob=np.concatenate((y_pred_prob,np.zeros(shape=(len(y_pred_prob),1))),1)
    #y_pred = model['rf_model'].predict(X_test_new);
    #y_pred_prob = model['rf_model'].predict_proba(X_test_new);
    # Naive accuracy (correct classification rate):
    accuracy = np.mean(y_pred == y);
    
    # Count occorrences of true-positive, true-negative, false-positive, and false-negative:
    tp = np.sum(np.logical_and(y_pred,y));
    tn = np.sum(np.logical_and(np.logical_not(y_pred),np.logical_not(y)));
    fp = np.sum(np.logical_and(y_pred,np.logical_not(y)));
    fn = np.sum(np.logical_and(np.logical_not(y_pred),y));
    
    sensitivity = 0;
    specificity = 0;
    precision = 0;
    
    # Sensitivity (=recall=true positive rate) and Specificity (=true negative rate):
    if tp+fn != 0:
        sensitivity = float(tp) / (tp+fn);
    if tn+fp != 0:
        specificity = float(tn) / (tn+fp);
    
    # Balanced accuracy is a more fair replacement for the naive accuracy:
    balanced_accuracy = (sensitivity + specificity) / 2.;
    
    # Precision:
    # Beware from this metric, since it may be too sensitive to rare labels.
    # In the ExtraSensory Dataset, there is large skew among the positive and negative classes,
    # and for each label the pos/neg ratio is different.
    # This can cause undesirable and misleading results when averaging precision across different labels.
    if tp+fp != 0:
        precision = float(tp) / (tp+fp);
    
    
    # Balanced accuracy is a more fair replacement for the naive accuracy:
    if (precision !=0) & (sensitivity!=0) :
        #F1_score_phone[i] [input_value]= 2 * (sensitivity * precision) / (sensitivity + precision)
        F1_score_phone = 2 * (sensitivity * precision) / (sensitivity + precision)

    
    print("-"*10);
    print('Accuracy*:         %.2f' % accuracy);
    print('Sensitivity (TPR): %.2f' % sensitivity);
    print('Specificity (TNR): %.2f' % specificity);
    print('Balanced accuracy: %.2f' % balanced_accuracy);
    print('Precision**:       %.2f' % precision);
    print("-"*10);
    print('F1 score**:       %.2f' % F1_score_phone);
    return(accuracy,balanced_accuracy,precision, y_pred_prob,  y);
    

  #%%
  #train part of mlp model
import sklearn.neural_network

def train_mlp_model(X_train,Y_train,M_train,feat_sensor_names,label_names,sensors_to_use,target_label):
    # Project the feature matrix to the features from the desired sensors:
    X_train = project_features_to_selected_sensors(X_train,feat_sensor_names,sensors_to_use);
    print("== Projected the features to %d features from the sensors: %s" % (X_train.shape[1],', '.join(sensors_to_use)));

    # It is recommended to standardize the features (subtract mean and divide by standard deviation),
    # so that all their values will be roughly in the same range:
    (mean_vec,std_vec) = estimate_standardization_params(X_train);
    X_train = standardize_features(X_train,mean_vec,std_vec);
    print (X_train.shape);
    # The single target label:
    label_ind = [label_names.index(label) for label in target_label];
    #label_ind = label_names.index(target_label);
    index_length = len(label_ind);
    y = Y_train[:,label_ind];
    missing_label = M_train[:,label_ind];
    existing_label = np.logical_not(missing_label);
    
    #Find all examples that are missing any of target labels:
    existing = existing_label[:,0];
    
    for i in range(1,index_length):
        existing = np.logical_and(existing,existing_label[:,i]);
    print (existing.shape);
    
    # Select only the examples that are not missing the target label:
    X_train = X_train[existing,:];
    y = y[existing];
    
    print (X_train.shape);
    print (y.shape);
    
    #If there is no example
    if len(y) == 0:
        mlp_model = None;
        # Assemble all the parts of the model:
        model = {            'sensors_to_use':sensors_to_use,            'target_label':target_label,            'mean_vec':mean_vec,            'std_vec':std_vec,            'mlp_model':mlp_model};
        return model;
    
    # Also, there may be missing sensor-features (represented in the data as NaN).
    # You can handle those by imputing a value of zero (since we standardized, this is equivalent to assuming average value).
    # You can also further select examples - only those that have values for all the features.
    # For this tutorial, let's use the simple heuristic of zero-imputation:
    X_train[np.isnan(X_train)] = 0.;
    
    #print("== Training with %d examples. For label '%s' we have %d positive and %d negative examples." % \
    #      (len(y),get_label_pretty_name(target_label),sum(y),sum(np.logical_not(y))) );
    
    # Typically, the data is highly imbalanced, with many more negative examples;
    # To avoid a trivial classifier (one that always declares 'no'), it is important to counter-balance the pos/neg classes:
    mlp_model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(8,8), max_iter=200, alpha=0.0001,                     solver='sgd', learning_rate_init=0.01, verbose= 1,                     tol=0.0001, momentum=0.8);
    mlp_model.fit(X_train,y);
    
    # Assemble all the parts of the model:
    model = {            'sensors_to_use':sensors_to_use,            'target_label':target_label,            'mean_vec':mean_vec,            'std_vec':std_vec,            'mlp_model':mlp_model};
    
    return model;
#%%
    #test part of mlp model
def test_mlp_model(X_test,Y_test,M_test,timestamps,feat_sensor_names,label_names,model):
    #Check if the model is null:
    if model['mlp_model'] == None:
        return (1.0,1.0,1.0);
    # Project the feature matrix to the features from the sensors that the classifier is based on:
    X_test = project_features_to_selected_sensors(X_test,feat_sensor_names,model['sensors_to_use']);
    print("== Projected the features to %d features from the sensors: %s" % (X_test.shape[1],', '.join(model['sensors_to_use'])));

    # We should standardize the features the same way the train data was standardized:
    X_test = standardize_features(X_test,model['mean_vec'],model['std_vec']);
    
    # The target label:
    target_label = model['target_label'];
    label_ind = [label_names.index(label) for label in target_label];
    #label_ind = label_names.index(model['target_label']);
    index_length = len(label_ind);
    y = Y_test[:,label_ind];
    missing_label = M_test[:,label_ind];
    existing_label = np.logical_not(missing_label);
    
    #Find all examples that are missing any of target labels:
    existing = existing_label[:,0];
    for i in range(1,index_length):
        existing = np.logical_and(existing,existing_label[:,i]);
    
    # Select only the examples that are not missing the target label:
    X_test = X_test[existing,:];
    y = y[existing];
    timestamps = timestamps[existing];
    
    print (X_test.shape);
    print (y.shape);
    
    #If there is no example
    if len(y) == 0:
        return (1.0,1.0,1.0);
    
    # Do the same treatment for missing features as done to the training data:
    X_test[np.isnan(X_test)] = 0.;
    
    print("== Testing with %d examples." % len(y));
    
    # Preform the prediction:
    y_pred = model['mlp_model'].predict(X_test);
    y_pred_prob = model['mlp_model'].predict_proba(X_test);
    
    # Naive accuracy (correct classification rate):
    accuracy = np.mean(y_pred == y);
    
    # Count occorrences of true-positive, true-negative, false-positive, and false-negative:
    tp = np.sum(np.logical_and(y_pred,y));
    tn = np.sum(np.logical_and(np.logical_not(y_pred),np.logical_not(y)));
    fp = np.sum(np.logical_and(y_pred,np.logical_not(y)));
    fn = np.sum(np.logical_and(np.logical_not(y_pred),y));
    
    sensitivity = 0;
    specificity = 0;
    precision = 0;
    
    # Sensitivity (=recall=true positive rate) and Specificity (=true negative rate):
    if tp+fn != 0:
        sensitivity = float(tp) / (tp+fn);
    if tn+fp != 0:
        specificity = float(tn) / (tn+fp);
    
    # Balanced accuracy is a more fair replacement for the naive accuracy:
    balanced_accuracy = (sensitivity + specificity) / 2.;
    
    # Precision:
    # Beware from this metric, since it may be too sensitive to rare labels.
    # In the ExtraSensory Dataset, there is large skew among the positive and negative classes,
    # and for each label the pos/neg ratio is different.
    # This can cause undesirable and misleading results when averaging precision across different labels.
    if tp+fp != 0:
        precision = float(tp) / (tp+fp);
    
    print("-"*10);
    print('Accuracy*:         %.2f' % accuracy);
    print('Sensitivity (TPR): %.2f' % sensitivity);
    print('Specificity (TNR): %.2f' % specificity);
    print('Balanced accuracy: %.2f' % balanced_accuracy);
    print('Precision**:       %.2f' % precision);
    print("-"*10);
    
    return(accuracy,balanced_accuracy,precision, y_pred_prob,  y);

#train part of LR model
def train_lr_model(X_train,Y_train,M_train,feat_sensor_names,label_names,sensors_to_use,target_label):
    # Project the feature matrix to the features from the desired sensors:
    X_train = project_features_to_selected_sensors(X_train,feat_sensor_names,sensors_to_use);
    print("== Projected the features to %d features from the sensors: %s" % (X_train.shape[1],', '.join(sensors_to_use)));

    # It is recommended to standardize the features (subtract mean and divide by standard deviation),
    # so that all their values will be roughly in the same range:
    (mean_vec,std_vec) = estimate_standardization_params(X_train);
    X_train = standardize_features(X_train,mean_vec,std_vec);
    print (X_train.shape)
    # The single target label:
    label_ind = label_names.index(target_label);
    y = Y_train[:,label_ind];
    missing_label = M_train[:,label_ind];
    existing_label = np.logical_not(missing_label);
    
    # Select only the examples that are not missing the target label:
    X_train = X_train[existing_label,:];
    y = y[existing_label];
    
    print( X_train.shape)
    print (y.shape)
    
    #If there is no example
    if len(y) == 0:
        lr_model = None;
        # Assemble all the parts of the model:
        model = {            'sensors_to_use':sensors_to_use,            'target_label':target_label,            'mean_vec':mean_vec,            'std_vec':std_vec,            'lr_model':lr_model};
        return model;
    
    # Also, there may be missing sensor-features (represented in the data as NaN).
    # You can handle those by imputing a value of zero (since we standardized, this is equivalent to assuming average value).
    # You can also further select examples - only those that have values for all the features.
    # For this tutorial, let's use the simple heuristic of zero-imputation:
    #X_train[np.isnan(X_train)] = 0.;
    y = y[~np.isnan(X_train).any(axis=1)]
    X_train= X_train[~np.isnan(X_train).any(axis=1)]
    print("== Training with %d examples. For label '%s' we have %d positive and %d negative examples." %           (len(y),get_label_pretty_name(target_label),sum(y),sum(np.logical_not(y))) );
    
    # Now, we have the input features and the ground truth for the output label.
    # We can train a logistic regression model.
    
    # Typically, the data is highly imbalanced, with many more negative examples;
    # To avoid a trivial classifier (one that always declares 'no'), it is important to counter-balance the pos/neg classes:
      #if sum(y) < sum(np.logical_not(y)):
    i_class0 = np.where(y == 1)[0]
    i_class1 = np.where(y == 0)[0]
    #else:
     #   i_class0 = np.where(y == 0)[0]
      #  i_class1 = np.where(y == 1)[0]
# Number of observations in each class
    n_class0 = len(i_class0)
    n_class1 = len(i_class1)

# For every observation in class 1, randomly sample from class 0 with replacement
    i_class0_upsampled = np.random.choice(i_class0, size=n_class1, replace=True)

# Join together class 0's upsampled target vector with class 1's target vector
    y=np.concatenate((y[i_class0_upsampled], y[i_class1]))
    X_train=np.concatenate((X_train[i_class0_upsampled], X_train[i_class1]))
    lr_model = sklearn.linear_model.LogisticRegression(random_state=0, class_weight='balanced', solver='lbfgs');
    lr_model.fit(X_train,y);
    print("== Training with %d examples. For label '%s' we have %d positive and %d negative examples." %           (len(y),get_label_pretty_name(target_label),sum(y),sum(np.logical_not(y))) );
   
    # Assemble all the parts of the model:
    model = {            'sensors_to_use':sensors_to_use,            'target_label':target_label,            'mean_vec':mean_vec,            'std_vec':std_vec,            'lr_model':lr_model};
    
    return model;




# In[26]:


#test part for LR model
def test_lr_model(X_test,Y_test,M_test,timestamps,feat_sensor_names,label_names,model):
    #Check if the model is null:
    if model['lr_model'] == None:
        return (1.0,1.0,1.0);
    
    # Project the feature matrix to the features from the sensors that the classifier is based on:
    X_test = project_features_to_selected_sensors(X_test,feat_sensor_names,model['sensors_to_use']);
    print("== Projected the features to %d features from the sensors: %s" % (X_test.shape[1],', '.join(model['sensors_to_use'])));

    # We should standardize the features the same way the train data was standardized:
    X_test = standardize_features(X_test,model['mean_vec'],model['std_vec']);
    
    # The single target label:
    label_ind = label_names.index(model['target_label']);
    y = Y_test[:,label_ind];
    missing_label = M_test[:,label_ind];
    existing_label = np.logical_not(missing_label);
    
    # Select only the examples that are not missing the target label:
    X_test = X_test[existing_label,:];
    y = y[existing_label];
    timestamps = timestamps[existing_label];
    
    print (X_test.shape);
    print (y.shape);
    
    #If there is no example
    if len(y) == 0:
        return (1.0,1.0,1.0);
    
    # Do the same treatment for missing features as done to the training data:
    #X_test[np.isnan(X_test)] = 0.;
    y = y[~np.isnan(X_test).any(axis=1)]
    X_test= X_test[~np.isnan(X_test).any(axis=1)]
    
    print("== Testing with %d examples. For label '%s' we have %d positive and %d negative examples." %           (len(y),get_label_pretty_name(target_label),sum(y),sum(np.logical_not(y))) );
    
    # Preform the prediction:
    y_pred = model['lr_model'].predict(X_test);
    y_pred_prob = model['lr_model'].predict_proba(X_test);
    
    
    # Naive accuracy (correct classification rate):
    accuracy = np.mean(y_pred == y);
    
    # Count occorrences of true-positive, true-negative, false-positive, and false-negative:
    tp = np.sum(np.logical_and(y_pred,y));
    tn = np.sum(np.logical_and(np.logical_not(y_pred),np.logical_not(y)));
    fp = np.sum(np.logical_and(y_pred,np.logical_not(y)));
    fn = np.sum(np.logical_and(np.logical_not(y_pred),y));
    
    sensitivity = 0;
    specificity = 0;
    precision = 0;
    
    # Sensitivity (=recall=true positive rate) and Specificity (=true negative rate):
    if tp+fn != 0:
        sensitivity = float(tp) / (tp+fn);
    if tn+fp != 0:
        specificity = float(tn) / (tn+fp);
    
    # Balanced accuracy is a more fair replacement for the naive accuracy:
    balanced_accuracy = (sensitivity + specificity) / 2.;
    
    # Precision:
    # Beware from this metric, since it may be too sensitive to rare labels.
    # In the ExtraSensory Dataset, there is large skew among the positive and negative classes,
    # and for each label the pos/neg ratio is different.
    # This can cause undesirable and misleading results when averaging precision across different labels.
    if tp+fp != 0:
        precision = float(tp) / (tp+fp);
          # Balanced accuracy is a more fair replacement for the naive accuracy:
    if (precision !=0) & (sensitivity!=0) :
        #F1_score_phone[i] [input_value]= 2 * (sensitivity * precision) / (sensitivity + precision)
        F1_score_phone = 2 * (sensitivity * precision) / (sensitivity + precision)
    
    print("-"*10);
    print('Accuracy*:         %.2f' % accuracy);
    print('Sensitivity (TPR): %.2f' % sensitivity);
    print('Specificity (TNR): %.2f' % specificity);
    print('Balanced accuracy: %.2f' % balanced_accuracy);
    print('Precision**:       %.2f' % precision);
    print("-"*10);
    print('F1 score**:       %.2f' % F1_score_phone);
    #return(accuracy,balanced_accuracy,precision);
    return (accuracy,balanced_accuracy,precision, y_pred_prob,  y);
    #%% 
txtname = open('/home/abraham/Documents/Rotation_two/Existing_python_implementation/all_uuids_new.txt','r');
txtdata = txtname.read().splitlines();
txtname.close();
print (len(txtdata))
#%%

#%%
def split_train_test_uuids(i,txtdata):
    i=0
    #test_uuid = txtdata[i];
    #train_uuid = [data for j,data in enumerate(txtdata) if j != i];
    test_uuid = [data for j,data in enumerate(txtdata) if j != i];
    train_uuid = [data for j,data in enumerate(txtdata) if j != i];
    return (test_uuid,train_uuid);


#%%
# Get train and test dataset
def unison_shuffled_copies(a, b,c,d,e,f):
    assert (len(a) == len(b))
    p = numpy.random.permutation(len(a))
    return a[p], b[p],c[p],d[p],e[p],f[p]
def randomize(a, b,c,d):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    shuffled_c = c[permutation]
    shuffled_d = d[permutation]
   
    return shuffled_a, shuffled_b,shuffled_c,shuffled_d
# In[68]:



def get_train_test_set(test_uuid,train_uuid):
    uuid = train_uuid[0];
    (train_X,train_Y,train_M,train_time,train_feature,train_label) = read_user_data(uuid);
    train_X_new,train_Y_new,train_M_new,train_time_new,train_feature_new,train_label_new = read_user_data(uuid);
    #COMMENTED BY ME
    #for i in range(1,58):
    for input_set in range(10,50):
        uuid=train_uuid[input_set];
        #print("the input set is %s:"% uuid)
        (X2,Y2,M2,timestamps2,feature_names2,label_names2) = read_user_data(uuid);
        train_X = np.concatenate((train_X,X2),axis=0);
        train_Y = np.concatenate((train_Y,Y2),axis=0);
        train_M = np.concatenate((train_M,M2),axis=0);
        train_time = np.concatenate((train_time,timestamps2),axis=0);
        train_feature = np.concatenate((train_feature,feature_names2),axis=0);
        train_label = np.concatenate((train_label,label_names2),axis=0);
    
   # (train_X,train_Y,train_M,train_time)=randomize(train_X,train_Y,train_M,train_time)
    
    test_len=3*math.floor(len(train_X)/10)  
    
    test_X=train_X[0:test_len,:]
    train_X_new=train_X[test_len : len(train_X),:]
   
    test_Y=train_Y[0:test_len,:]
    train_Y_new=train_Y[test_len:len(train_Y),:]
    
    test_M=train_M[0:test_len,:]
    train_M_new=train_M[test_len:len(train_M),:]
    
    test_time=train_time[0:test_len]
    train_time_new=train_time[test_len:len(train_time)]
    
    test_feature=train_feature[0:3*math.floor(len(train_feature)/10)]
    train_feature_new=train_feature[3*math.floor(len(train_feature)/10): len(train_feature)]
    #test_feature=train_feature
    #train_feature_new=train_feature
    
    test_label=train_label[0:3*math.floor(len(train_label)/10) ]
    train_label_new=train_label[3*math.floor(len(train_label)/10):len(train_label)]
    
    #test_label=train_label
    #train_label_new=train_label
   
    
   # uuid = test_uuid[0];
    #(test_X,test_Y,test_M,test_time,test_feature,test_label) = read_user_data(uuid);
    #for test_set in range(1,10):
     #   uuid=test_uuid[test_set];
      #  (X3,Y3,M3,timestamps3,feature_names3,label_names3) = read_user_data(uuid);
       # test_X = np.concatenate((test_X,X3),axis=0);
        #test_Y = np.concatenate((test_Y,Y3),axis=0);
        #test_M = np.concatenate((test_M,M3),axis=0);
        #test_time = np.concatenate((test_time,timestamps3),axis=0);
        #test_feature = np.concatenate((test_feature,feature_names3),axis=0);
        #test_label = np.concatenate((test_label,label_names3),axis=0);
        
    #return (train_X,train_Y,train_M,train_time,train_feature,train_label, test_X,test_Y,test_M,test_time,test_feature,test_label);
    return (train_X_new,train_Y_new,train_M_new,train_time_new,train_feature_new,train_label_new, test_X,test_Y,test_M,test_time,test_feature,test_label);


# For LR Model:

# In[116]:


#sensor_set = ['Acc','Gyro','Magnet']
#target_label = 'PHONE_IN_HAND';
#target_label = 'LYING_DOWN';
#sensors_to_use = ['Acc','Gyro','Magnet'];
#y_prob= np.zeros(60)
#y_label=np.zeros(60)
#acc = np.zeros(60);
#ba = np.zeros(60);
#precise = np.zeros(60);
#for i in range(60)
#for i in range(0,1): 
sensors_to_use = ['Acc'];
#sensors_to_use = ['Acc','Gyro','Magnet'];
#sensors_to_use = ['Acc','Gyro','Magnet','loc'];
#sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud'];
#sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud','WAcc','Compass'];
target_label = 'PHONE_IN_HAND'; 
i=3
(test_uuid,train_uuid) = split_train_test_uuids(i,txtdata);
(train_X,train_Y,train_M,train_time,train_feature,train_label,     test_X,test_Y,test_M,test_time,test_feature,test_label) =     get_train_test_set(test_uuid,train_uuid);
#print test_X.shape;
#model_hand = train_lr_model(train_X,train_Y,train_M,feat_sensor_names, label_names,sensors_to_use,target_label);
model_hand = train_rf_model(train_X,train_Y,train_M,feat_sensor_names, label_names,sensors_to_use,target_label);
#(acc[i],ba[i],precise[i],y_prob[i],y_label[i]) =     test_rf_model(test_X,test_Y,test_M,test_time,feat_sensor_names,label_names,model);
#(acc,ba,precise,y_prob_hand,y_label_hand) =     test_lr_model(test_X,test_Y,test_M,test_time,feat_sensor_names,label_names,model_hand);
(acc,ba,precise,y_prob_hand,y_label_hand) =     test_rf_model(test_X,test_Y,test_M,test_time,feat_sensor_names,label_names,model_hand);


#%%#%%#%%#%%

#%%
#%%#%%#%%#%%
sensors_to_use = ['Acc'];
#sensors_to_use = ['Acc','Gyro','Magnet'];
#sensors_to_use = ['Acc','Gyro','Magnet','loc'];
#sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud'];
#sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud','WAcc','Compass'];
target_label = 'PHONE_IN_BAG';
#model_BAG = train_rf_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
#model = train_svm_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
#print model
i=3
(test_uuid,train_uuid) = split_train_test_uuids(i,txtdata);
(train_X,train_Y,train_M,train_time,train_feature,train_label,     test_X,test_Y,test_M,test_time,test_feature,test_label) =     get_train_test_set(test_uuid,train_uuid);
#print test_X.shape;
#model_BAG = train_lr_model(train_X,train_Y,train_M,feat_sensor_names, label_names,sensors_to_use,target_label);
model_BAG = train_rf_model(train_X,train_Y,train_M,feat_sensor_names, label_names,sensors_to_use,target_label);
#(acc,ba,precise,y_prob_Bag,y_label_Bag) =     test_rf_model(test_X,test_Y,test_M,test_time,feat_sensor_names,label_names,model_BAG);
#(acc,ba,precise,y_prob_Bag,y_label_BAG) =     test_lr_model(test_X,test_Y,test_M,test_time,feat_sensor_names,label_names,model_BAG);
(acc,ba,precise,y_prob_Bag,y_label_BAG) =  test_rf_model(test_X,test_Y,test_M,test_time,feat_sensor_names,label_names,model_BAG);
#%%#%%#%%#%%#%%
sensors_to_use = ['Acc'];
#sensors_to_use = ['Acc','Gyro','Magnet'];
#sensors_to_use = ['Acc','Gyro','Magnet','loc'];
#sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud'];
#sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud','WAcc','Compass'];
target_label = 'PHONE_ON_TABLE';
#model = train_svm_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
#model_table =train_rf_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
#print model
i=3
(test_uuid,train_uuid) = split_train_test_uuids(i,txtdata);
(train_X,train_Y,train_M,train_time,train_feature,train_label,     test_X,test_Y,test_M,test_time,test_feature,test_label) =     get_train_test_set(test_uuid,train_uuid);
#print test_X.shape;
#model_table = train_lr_model(train_X,train_Y,train_M,feat_sensor_names, label_names,sensors_to_use,target_label);
model_table = train_rf_model(train_X,train_Y,train_M,feat_sensor_names, label_names,sensors_to_use,target_label);
#(acc[i],ba[i],precise[i],y_prob[i],y_label[i]) =     test_rf_model(test_X,test_Y,test_M,test_time,feat_sensor_names,label_names,model);
#(acc,ba,precise,y_prob_TABLE,y_label_TABLE) =     test_lr_model(test_X,test_Y,test_M,test_time,feat_sensor_names,label_names,model_table);
(acc,ba,precise,y_prob_TABLE,y_label_TABLE) =  test_rf_model(test_X,test_Y,test_M,test_time,feat_sensor_names,label_names,model_table);
#%%
#%%#%%#%%#%%#%%
sensors_to_use = ['Acc'];
#sensors_to_use = ['Acc','Gyro','Magnet'];
#sensors_to_use = ['Acc','Gyro','Magnet','loc'];
#sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud'];
#sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud','WAcc','Compass'];
target_label = 'PHONE_IN_POCKET';
#model = train_svm_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
#model_pocket = train_rf_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
#print model
i=3
(test_uuid,train_uuid) = split_train_test_uuids(i,txtdata);
(train_X,train_Y,train_M,train_time,train_feature,train_label,     test_X,test_Y,test_M,test_time,test_feature,test_label) =     get_train_test_set(test_uuid,train_uuid);
#print test_X.shape;
#model_pocket = train_lr_model(train_X,train_Y,train_M,feat_sensor_names, label_names,sensors_to_use,target_label);
model_pocket = train_rf_model(train_X,train_Y,train_M,feat_sensor_names, label_names,sensors_to_use,target_label);
#(acc[i],ba[i],precise[i],y_prob[i],y_label[i]) =     test_rf_model(test_X,test_Y,test_M,test_time,feat_sensor_names,label_names,model);
#(acc,ba,precise,y_prob_POCKET,y_label_POCKET) =     test_lr_model(test_X,test_Y,test_M,test_time,feat_sensor_names,label_names,model_pocket);
(acc,ba,precise,y_prob_POCKET,y_label_POCKET) = test_rf_model(test_X,test_Y,test_M,test_time,feat_sensor_names,label_names,model_pocket);
#
# In[40]:

#print ("The length of indoor lochome and meeting are %s %s %s "% (len(y_prob_indoors), len(y_prob_lochome), len(y_prob_meeting)))                                                                                                                                                                                                                                                                                                                                                                                                

length_min_phone=min(len(y_prob_POCKET),len(y_prob_TABLE),len(y_prob_hand),len(y_prob_Bag))

y_prob_phone_pocket=np.zeros(shape=(length_min_phone,1))
y_prob_phone_table=np.zeros(shape=(length_min_phone,1))
y_prob_phone_hand=np.zeros(shape=(length_min_phone,1))
y_prob_phone_BAG=np.zeros(shape=(length_min_phone,1))   

y_comp_phone_pocket=np.zeros(shape=(length_min_phone,1))
y_comp_phone_table=np.zeros(shape=(length_min_phone,1))
y_comp_phone_hand=np.zeros(shape=(length_min_phone,1))
y_comp_phone_BAG=np.zeros(shape=(length_min_phone,1))

y_pred_phone_pocket=np.zeros(shape=(length_min_phone,1))
y_pred_phone_table=np.zeros(shape=(length_min_phone,1))
y_pred_phone_hand=np.zeros(shape=(length_min_phone,1))
y_pred_phone_BAG=np.zeros(shape=(length_min_phone,1))
   
#sum_test=0       
#change the existing label from boealean to number
for i in range(0,length_min_phone):
    if (y_label_hand[i]==True):
        y_pred_phone_hand[i] = 1
        
    else:
        y_pred_phone_hand[i] = 0 
        
    if (y_label_TABLE [i]==True):
        y_pred_phone_table[i] = 1
      
    else:
        y_pred_phone_table[i] = 0   
        
    if (y_label_POCKET [i]==True):
        y_pred_phone_pocket[i] = 1
        
    else:
        y_pred_phone_pocket[i] = 0 
        
    if (y_label_BAG [i]==True):
        y_pred_phone_BAG[i] = 1
       # sum_test=y_pred_phone_BAG[i] + sum_test
    else:
        y_pred_phone_BAG[i] = 0   
     
#finding probabilities of phone location
for i in range (0,length_min_phone):
    y_prob_phone_pocket[i]= y_prob_POCKET[i,1]
    y_prob_phone_table[i]=  y_prob_TABLE [i,1]
    y_prob_phone_hand[i]=   y_prob_hand [i,1]
    y_prob_phone_BAG[i] =   y_prob_Bag [i,1]
    
        #        
#comparingphone probabilities phone location
for i in range (0, length_min_phone):
    max_value=max(y_prob_phone_pocket[i],y_prob_phone_table[i],y_prob_phone_hand[i],y_prob_phone_BAG[i])
    if (y_prob_phone_pocket[i] == max_value):
        y_comp_phone_pocket[i] = 1
        y_comp_phone_table[i] = 0
        y_comp_phone_hand[i]= 0
        y_comp_phone_BAG[i]= 0
       # y_comp_indoor_gym[i] =0
    if (y_prob_phone_table[i] == max_value):
    #elif (y_prob_phone_table[i] == max_value):
        y_comp_phone_pocket[i] = 0
        y_comp_phone_table[i] = 1
        y_comp_phone_hand[i]=0
        y_comp_phone_BAG[i]=0
        #y_prob_indoor_gym[i] =0
    #elif (y_prob_phone_hand[i] == max_value):
    if (y_prob_phone_hand[i] == max_value):
    #else:
        y_comp_phone_pocket[i] = 0
        y_comp_phone_table[i] = 0
        y_comp_phone_hand[i]= 1
        y_comp_phone_BAG[i]= 0
        
    if (y_prob_phone_BAG[i] == max_value):
    #else:
        y_comp_phone_pocket[i] = 0
        y_comp_phone_table[i] = 0
        y_comp_phone_hand[i]=0
        y_comp_phone_BAG[i]=1 

   #%%              #%%

   
y_pred_phone=np.zeros(shape=(length_min_phone,1))
y_orig_phone=np.zeros(shape=(length_min_phone,1))
   # F1_score_phone=np.zeros(shape=(cont_num,UUID_length))

 #%%   
#pr
#find the accuracy and balanced accuracy for phone location
for i in range(0,cont_num):
    if i==0:
        y_pred_phone=y_comp_phone_pocket
        y_orig_phone=y_pred_phone_pocket
        string_value="phone pocket "
    if i==1:
    #if i==1:
        y_pred_phone=y_comp_phone_table
        y_orig_phone=y_pred_phone_table
        string_value="phone table  "
    if i==2:
    #else:
        y_pred_phone=y_comp_phone_hand
        y_orig_phone=y_pred_phone_hand
        string_value="phone on hand"
    if i==3:
    #else:
        y_pred_phone=y_comp_phone_BAG
        y_orig_phone=y_pred_phone_BAG
        string_value="phone on BAG"
        
    tp = np.sum(np.logical_and(y_pred_phone,y_orig_phone));
    tn = np.sum(np.logical_and(np.logical_not(y_pred_phone),np.logical_not(y_orig_phone)));
    fp = np.sum(np.logical_and(y_pred_phone,np.logical_not(y_orig_phone)));
    fn = np.sum(np.logical_and(np.logical_not(y_pred_phone),y_orig_phone));
    sensitivity = 0;
    specificity = 0;
    precision = 0;
    
    # Sensitivity (=recall=true positive rate) and Specificity (=true negative rate):
    if tp+fn != 0:
        sensitivity = float(tp) / (tp+fn);
    if tn+fp != 0:
        specificity = float(tn) / (tn+fp);
    if tp+fp != 0:
        precision= float(tp)/(tp+fp);
    
    # Balanced accuracy is a more fair replacement for the naive accuracy:
    if (precision !=0) & (sensitivity!=0) :
        #F1_score_phone[i] [input_value]= 2 * (sensitivity * precision) / (sensitivity + precision)
        F1_score_phone[i] = 2 * (sensitivity * precision) / (sensitivity + precision)
    # Balanced accuracy is a more fair replacement for the naive accuracy:
    #balanced_accuracy_phone[i][input_value]=(sensitivity + specificity) / 2.;
    #accuracy_phone[i][input_value]=np.mean(y_pred_phone==y_orig_phone);
   # print('the accuracy, balanced accuracy, F1 score of %s is : %f %f %f '% (string_value, accuracy_phone[i],balanced_accuracy_phone[i],F1_score_phone[i]))
    print('the F1 score of %s is :  %f '% (string_value, F1_score_phone[i]))
   
