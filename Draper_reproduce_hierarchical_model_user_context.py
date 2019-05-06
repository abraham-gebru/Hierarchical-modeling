#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 00:52:39 2019

@author: abraham
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 03:26:45 2019

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
UUID_length= 13
cont_num=2
count_user_act=6
count_second_level=5
input_init=2

F1_score_act_inactive=np.zeros(shape=(cont_num,(UUID_length)))
F1_score_user=np.zeros(shape=(count_user_act,(UUID_length)))
F1_score_second_level=np.zeros(shape=(count_second_level,(UUID_length)))
balanced_accuracy_phone= np.zeros(shape=(cont_num,UUID_length))
accuracy_phone=np.zeros(shape=(cont_num,UUID_length))
    
for input_value in range(input_init,UUID_length):
  
    if input_value==1:
        uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/0BFC35E2-4817-4865-BFA7-764742302A2D';
    #elif input_value==2:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/abraham-1155FF54-63D3-4AB2-9863-8385D0BD0A13';
    #elif input_value==3:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/abraham-11B5EC4D-4133-4289-B475-4E737182A406';
   # elif input_value==4:
    #    uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/0BFC35E2-4817-4865-BFA7-764742302A2D';
    #elif input_value==5:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/0E6184E1-90C0-48EE-B25A-F1ECB7B9714E';
   # elif input_value==6:
    #    uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842';
   # elif input_value==7:
    elif input_value==2:    
        uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/2C32C23E-E30C-498A-8DD2-0EFB9150A02E'; 
    #elif input_value==8:    
    elif input_value==3:
        uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/abrahamDRAPER-11B5EC4D-4133-4289-B475-4E737182A406'; 
    #elif input_value==9:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/4E98F91F-4654-42EF-B908-A3389443F2E7';
    #elif input_value==10:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/5EF64122-B513-46AE-BCF1-E62AAC285D2C';
    #elif input_value==4:
   # elif input_value==11:    
    #    uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/7CE37510-56D0-4120-A1CF-0E23351428D2';
   # elif input_value==12:
    #    uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/7D9BB102-A612-4E2A-8E22-3159752F55D8';
    #elif input_value==13:    
    elif input_value==4:    
        uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/9DC38D04-E82E-4F29-AB52-B476535226F2';
    #elif input_value==14:
    elif input_value==5:    
        uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/11B5EC4D-4133-4289-B475-4E737182A406';
    #elif input_value==15:
    elif input_value==6:    
        uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/11B5EC4D-4133-4289-B475-4E737182A406';
    #elif input_value==16:
    elif input_value==7:    
        uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/24E40C4C-A349-4F9F-93AB-01D00FB994AF';
   # elif input_value==17:
    #    uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/27E04243-B138-4F40-A164-F40B60165CF3';
    #elif input_value==18:
    elif input_value==8:
        uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/33A85C34-CFE4-4732-9E73-0A7AC861B27A';
   # elif input_value==19:    
    #    uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/40E170A7-607B-4578-AF04-F021C3B0384A';
    #elif input_value==20:
    #elif input_value==5:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/59EEFAE0-DEB0-4FFF-9250-54D2A03D0CF2';
    #elif input_value==21:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/74B86067-5D4B-43CF-82CF-341B76BEA0F4';
    #elif input_value==22:
    elif input_value==9:    
        uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/78A91A4E-4A51-4065-BDA7-94755F0BB3BB';
    #elif input_value==23:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/83CF687B-7CEC-434B-9FE8-00C3D5799BE6';
   # elif input_value==24:
   # elif input_value==6:    
    #    uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/86A4F379-B305-473D-9D83-FC7D800180EF';
    #elif input_value==25:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/96A358A0-FFF2-4239-B93E-C7425B901B47';
    #elif input_value==26:
    elif input_value==10:
        uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/098A72A5-E3E5-4F54-A152-BBDA0DF7B694';
   # elif input_value==27:
    #    uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/99B204C0-DD5C-4BB7-83E8-A37281B8D769';
    #elif input_value==28:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/481F4DD2-7689-43B9-A2AA-C8772227162B';
   # elif input_value==29:
    #    uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/797D145F-3858-4A7F-A7C2-A4EB721E133C';
    #elif input_value==30:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/1155FF54-63D3-4AB2-9863-8385D0BD0A13';
    #elif input_value==31:    
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/1538C99F-BA1E-4EFB-A949-6C7C47701B20';
    #elif input_value==32:
    #elif input_value==7:    
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/3600D531-0C55-44A7-AE95-A7A38519464E';
   # elif input_value==33:
    #elif input_value==8:    
    #    uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/5119D0F8-FCA8-4184-A4EB-19421A40DE0D';
    #elif input_value==34:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/5152A2DF-FAF3-4BA8-9CA9-E66B32671A53';
    #elif input_value==36:
    elif input_value==11:    
        uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/8023FE1A-D3B0-4E2C-A57A-9321B7FC755F';
    #elif input_value==37:    
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/59818CD2-24D7-4D32-B133-24C2FE3801E5';
    #elif input_value==38:
    #elif input_value==10:    
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/61976C24-1C50-4355-9C49-AAE44A7D09F6';
    #elif input_value==39:
    elif input_value==12:
        uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/81536B0A-8DBF-4D8A-AC24-9543E2E4C8E0';
   # elif input_value==40:
    #    uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/136562B6-95B2-483D-88DC-065F28409FD2';
   # elif input_value==41:
    #    uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/665514DE-49DC-421F-8DCB-145D0B2609AD';
    #elif input_value==18:
    #elif input_value==42:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/806289BC-AD52-4CC1-806C-0CDB14D65EB6';
    #elif input_value==43:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/9759096F-1119-4E19-A0AD-6F16989C7E1C';
   # elif input_value==44:    
    #    uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/61359772-D8D8-480D-B623-7C636EAD0C81';
    #elif input_value==45:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/A5A30F76-581E-4757-97A2-957553A2C6AA';
    #elif input_value==46:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/A5CDF89D-02A2-4EC1-89F8-F534FDABDD96';
    #elif input_value==48:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/A76A5AF5-5A93-4CF2-A16E-62353BB70E8A';
   # elif input_value==49:
    #    uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/A7599A50-24AE-46A6-8EA6-2576F1011D81';
    #elif input_value==50:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/abraham-1155FF54-63D3-4AB2-9863-8385D0BD0A13';
    #elif input_value==51:
    #elif input_value==12:    
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/B7F9D634-263E-4A97-87F9-6FFB4DDCB36C';
    #elif input_value==53:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/B09E373F-8A54-44C8-895B-0039390B859F';
    #elif input_value==54:
    #elif input_value==13:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/B9724848-C7E2-45F4-9B3F-A1F38D864495';
   # elif input_value==56:
    #    uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/BE3CA5A6-A561-4BBD-B7C9-5DF6805400FC';
   # elif input_value==57:    
    #    uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/BEF6C611-50DA-4971-A040-87FB979F3FC1';
    #elif input_value==58:
    #elif input_value==10:
    #elif input_value==6:    
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/C48CE857-A0DD-4DDB-BEA5-3A25449B2153';
    #elif input_value==59:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/CA820D43-E5E2-42EF-9798-BE56F776370B';
    #elif input_value==60:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/CCAF77F0-FABB-4F2F-9E24-D56AD0C5A82F';
    #elif input_value==61:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/CDA3BBF7-6631-45E8-85BA-EEB416B32A3C';
    #elif input_value==30:
    #elif input_value==63:
    #elif input_value==15:
    #    uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/CF722AA9-2533-4E51-9FEB-9EAC84EE9AAC';
    #elif input_value==31:
    #elif input_value==16:    
    #elif input_value==64:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/D7D20E2E-FC78-405D-B346-DBD3FD8FC92B';
    #elif input_value==65:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/E65577C1-8D5D-4F70-AF23-B3ADB9D3DBA3';
    #elif input_value==66:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/F50235E0-DD67-4F2A-B00B-1F31ADA998B9';
    #elif input_value==67:
     #   uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/FDAA70A1-42A3-4E3F-9AE3-3FDA412E03BF';
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
#fig = plt.figure(figsize=(15,5),facecolor='white');

#ax1 = plt.subplot(1,2,1);
#labels_to_display = ['LYING_DOWN','SITTING','OR_standing','FIX_walking','FIX_running'];
#figure__pie_chart(Y,label_names,labels_to_display,'Body state',ax1);

#ax2 = plt.subplot(1,2,2);
#labels_to_display = ['PHONE_IN_HAND','PHONE_IN_BAG','PHONE_IN_POCKET','PHONE_ON_TABLE'];
#figure__pie_chart(Y,label_names,labels_to_display,'Phone position',ax2);


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
    #
    
    # In[12]:
    
    
    #figure__context_over_participation_time(timestamps,Y,label_names,labels_to_display,label_colors,use_actual_dates=True);
    
    
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
    
    #fig = plt.figure(figsize=(15,15),facecolor='white');
    #ax = plt.subplot(1,1,1);
    #plt.imshow(J,interpolation='none');plt.colorbar();
    
    #pretty_label_names = [get_label_pretty_name(label) for label in label_names];
    #n_labels = len(label_names);
    #ax.set_xticks(range(n_labels));
    #ax.set_xticklabels(pretty_label_names,rotation=45,ha='right',fontsize=7);
    #ax.set_yticks(range(n_labels));
    #ax.set_yticklabels(pretty_label_names,fontsize=7);
    
    
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
            model = {            'sensors_to_use':sensors_to_use,            'target_label':target_label,            'mean_vec':mean_vec,            'std_vec':std_vec,            'rf_model':rf_model};
            return model;
        
        # Also, there may be missing sensor-features (represented in the data as NaN).
        # You can handle those by imputing a value of zero (since we standardized, this is equivalent to assuming average value).
        # You can also further select examples - only those that have values for all the features.
        # For this tutorial, let's use the simple heuristic of zero-imputation:
       # X_train[np.isnan(X_train)] = 0.;
       # X_train.dropna(inplace=True)
       # X_train[np.isnan(X_train)] = 0.;
        y = y[~np.isnan(X_train).any(axis=1)]
        X_train= X_train[~np.isnan(X_train).any(axis=1)]
        
        #X_train_new = X_train[~np.isnan(X_train)]
        
        #X_train_new=X_train(~np.isnan(X_train))
        print("== Training with %d examples. For label '%s' we have %d positive and %d negative examples." %           (len(y),get_label_pretty_name(target_label),sum(y),sum(np.logical_not(y))) );
        
        # Now, we have the input features and the ground truth for the output label.
        # We can train a logistic regression model.
        
        # Typically, the data is highly imbalanced, with many more negative examples;
        # To avoid a trivial classifier (one that always declares 'no'), it is important to counter-balance the pos/neg classes:
        rf_model = sklearn.ensemble.RandomForestClassifier(n_estimators=10,class_weight='balanced');
        rf_model.fit(X_train,y);
        #rf_model.fit(X_train_new,y_new);
        
        # Assemble all the parts of the model:
        model = {            'sensors_to_use':sensors_to_use,            'target_label':target_label,            'mean_vec':mean_vec,            'std_vec':std_vec,            'rf_model':rf_model};
        
        return model;
    
    
    # In[50]:
    
    
    # In[51]:
    
    
    #test part of random forest
    def test_rf_model(X_test,Y_test,M_test,timestamps,feat_sensor_names,label_names,model):
        #Check if the model is null:
        if model['rf_model'] == None:
            return (1.0,1.0,1.0,1.0,1.0);
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
        if sum(y_pred_prob[:,0]) == len(y_pred_prob):
            y_pred_prob=np.concatenate((y_pred_prob,np.zeros(shape=(len(y_pred_prob),1))),1)
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
        
        print("-"*10);
        print('Accuracy*:         %.2f' % accuracy);
        print('Sensitivity (TPR): %.2f' % sensitivity);
        print('Specificity (TNR): %.2f' % specificity);
        print('Balanced accuracy: %.2f' % balanced_accuracy);
        print('Precision**:       %.2f' % precision);
        print("-"*10);
        
        return(accuracy,balanced_accuracy,precision, y_pred_prob,  y);
        
    
  
    #%%
    #%%#%%#%%#%%
    #sensors_to_use = ['Acc','WAcc','Gyro','Magnet','Compass','Aud'];
    #sensors_to_use = ['Acc','Gyro','Magnet'];
    sensors_to_use = ['Acc','Gyro','Magnet','loc'];
    #sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud'];
    #sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud','WAcc','Compass'];
    target_label = 'Active';
    #model = train_svm_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
    model_active =train_rf_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
    #print model
    # In[40]:
    
    
    #uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/11B5EC4D-4133-4289-B475-4E737182A406';
    #(X2,Y2,M2,timestamps2,feature_names2,label_names2) = read_user_data(uuid);
    
    # All the user data files should have the exact same columns. We can validate it:
    #validate_column_names_are_consistent(feature_names,feature_names2);
    #validate_column_names_are_consistent(label_names,label_names2);
    
    
    # In[41]:
    
    
    #(a,b,p,y_prob_active,y_label_Active) = test_svm_model(X,Y,M,timestamps,feat_sensor_names,label_names,model);
    (a,b,p,y_prob_active,y_label_Active) = test_rf_model(X,Y,M,timestamps,feat_sensor_names,label_names,model_active);
    #%%
    #%%#%%#%%#%%
    #sensors_to_use = ['Acc','WAcc','Gyro','Magnet','Compass','Aud'];
    #sensors_to_use = ['Acc','Gyro','Magnet'];
    sensors_to_use = ['Acc','Gyro','Magnet','loc'];
   # sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud'];
    #sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud','WAcc','Compass'];
    target_label = 'InActive';
    model_InActive = train_rf_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
    #model = train_svm_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
    #print model
    # In[40]:
    
    
    #uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/11B5EC4D-4133-4289-B475-4E737182A406';
    #(X2,Y2,M2,timestamps2,feature_names2,label_names2) = read_user_data(uuid);
    
    # All the user data files should have the exact same columns. We can validate it:
    #validate_column_names_are_consistent(feature_names,feature_names2);
    #validate_column_names_are_consistent(label_names,label_names2);
    
    
    # In[41]:
    
    
    (a,b,p, y_prob_InActive, y_label_InActive) = test_rf_model(X,Y,M,timestamps,feat_sensor_names,label_names, model_InActive);
    #(a,b,p,y_prob_InActive,y_label_InActive) = test_svm_model(X,Y,M,timestamps,feat_sensor_names,label_names,model);
    #%%#%%#%%#%%#%%
    #sensors_to_use = ['Acc','WAcc','Gyro','Magnet','Compass','Aud'];
    #sensors_to_use = ['Acc','Gyro','Magnet'];
    sensors_to_use = ['Acc','Gyro','Magnet','loc'];
    #sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud'];
    #sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud','WAcc','Compass'];
    target_label = 'Transport';
    #model = train_svm_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
    model_transport =train_rf_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
    #print model
    # In[40]:
    
    
    #uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/11B5EC4D-4133-4289-B475-4E737182A406';
    #(X2,Y2,M2,timestamps2,feature_names2,label_names2) = read_user_data(uuid);
    
    # All the user data files should have the exact same columns. We can validate it:
    #validate_column_names_are_consistent(feature_names,feature_names2);
    #validate_column_names_are_consistent(label_names,label_names2);
    
    
    # In[41]:
    
    
    #(a,b,p,y_prop_Transport,y_label_Transport) = test_svm_model(X,Y,M,timestamps,feat_sensor_names,label_names,model);
    
    (a,b,p,y_prop_Transport,y_label_Transport) = test_rf_model(X,Y,M,timestamps,feat_sensor_names,label_names,model_transport);
    #%%
    #%%#%%#%%#%%#%%
    #sensors_to_use = ['Acc','WAcc','Gyro','Magnet','Compass','Aud'];
    #sensors_to_use = ['Acc','Gyro','Magnet'];
    sensors_to_use = ['Acc','Gyro','Magnet','loc'];
    #sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud'];
    #sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud','WAcc','Compass'];
    target_label = 'userprop';
    #model = train_svm_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
    model_userprop = train_rf_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
    #print model
    # In[40]:
    
    
    #uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/11B5EC4D-4133-4289-B475-4E737182A406';
    #(X2,Y2,M2,timestamps2,feature_names2,label_names2) = read_user_data(uuid);
    
    # All the user data files should have the exact same columns. We can validate it:
    #validate_column_names_are_consistent(feature_names,feature_names2);
    #validate_column_names_are_consistent(label_names,label_names2);
    
    
    # In[41]:
    
    
    #(a,b,p,y_prob_userprop,y_label_userprop) = test_svm_model(X,Y,M,timestamps,feat_sensor_names,label_names,model);
    (a,b,p,y_prob_userprop,y_label_userprop) = test_rf_model(X,Y,M,timestamps,feat_sensor_names,label_names,model_userprop);
    #%%
      #%%#%%#%%#%%#%%
    #sensors_to_use = ['Acc','WAcc','Gyro','Magnet','Compass','Aud'];
    #sensors_to_use = ['Acc','Gyro','Magnet'];
    sensors_to_use = ['Acc','Gyro','Magnet','loc'];
    #sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud'];
    #sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud','WAcc','Compass'];
    target_label = 'DRIVE_-_I_M_A_PASSENGER';
    #model = train_svm_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
    model_passenger = train_rf_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
    #print model
    # In[40]:
    
    
    #uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/11B5EC4D-4133-4289-B475-4E737182A406';
    #(X2,Y2,M2,timestamps2,feature_names2,label_names2) = read_user_data(uuid);
    
    # All the user data files should have the exact same columns. We can validate it:
    #validate_column_names_are_consistent(feature_names,feature_names2);
    #validate_column_names_are_consistent(label_names,label_names2);
    
    
    # In[41]:
    
    
    #(a,b,p,y_prob_passenger,y_label_passenger) = test_svm_model(X,Y,M,timestamps,feat_sensor_names,label_names,model);
    (a,b,p,y_prob_passenger,y_label_passenger) = test_rf_model(X,Y,M,timestamps,feat_sensor_names,label_names,model_passenger);
       #%%  #%%#%%#%%#%%#%%
    #sensors_to_use = ['Acc','WAcc','Gyro','Magnet','Compass','Aud'];
    #sensors_to_use = ['Acc','Gyro','Magnet'];
    sensors_to_use = ['Acc','Gyro','Magnet','loc'];
    #sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud'];
    #sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud','WAcc','Compass'];
    target_label = 'DRIVE_-_I_M_THE_DRIVER';
    #model = train_svm_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
    model_driver = train_rf_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
    #print model
    # In[40]:
    
    
    #uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/11B5EC4D-4133-4289-B475-4E737182A406';
    #(X2,Y2,M2,timestamps2,feature_names2,label_names2) = read_user_data(uuid);
    
    # All the user data files should have the exact same columns. We can validate it:
    #validate_column_names_are_consistent(feature_names,feature_names2);
    #validate_column_names_are_consistent(label_names,label_names2);
    
    
    # In[41]:
    
    
    #(a,b,p,y_prob_Driver,y_label_Driver) = test_svm_model(X,Y,M,timestamps,feat_sensor_names,label_names,model);
    (a,b,p,y_prob_Driver,y_label_Driver) = test_rf_model(X,Y,M,timestamps,feat_sensor_names,label_names,model_driver);
    
   #%%  #%%#%%#%%#%%#%%
    #sensors_to_use = ['Acc','WAcc','Gyro','Magnet','Compass','Aud'];
    #sensors_to_use = ['Acc','Gyro','Magnet'];
    sensors_to_use = ['Acc','Gyro','Magnet','loc'];
    #sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud'];
    #sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud','WAcc','Compass'];
    target_label = 'FIX_walking';
    #model = train_svm_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
    model_walking = train_rf_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
    #print model
    # In[40]:
    
    
    #uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/11B5EC4D-4133-4289-B475-4E737182A406';
    #(X2,Y2,M2,timestamps2,feature_names2,label_names2) = read_user_data(uuid);
    
    # All the user data files should have the exact same columns. We can validate it:
    #validate_column_names_are_consistent(feature_names,feature_names2);
    #validate_column_names_are_consistent(label_names,label_names2);
    
    
    # In[41]:
    
    
    #(a,b,p,y_prob_walking,y_label_walking) = test_svm_model(X,Y,M,timestamps,feat_sensor_names,label_names,model);
    (a,b,p,y_prob_walking,y_label_walking) = test_rf_model(X,Y,M,timestamps,feat_sensor_names,label_names,model_walking);
       #%%  #%%#%%#%%#%%#%%
    #sensors_to_use = ['Acc','WAcc','Gyro','Magnet','Compass','Aud'];
    #sensors_to_use = ['Acc','Gyro','Magnet'];
    sensors_to_use = ['Acc','Gyro','Magnet','loc'];
    #sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud'];
    #sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud','WAcc','Compass'];
    target_label = 'BICYCLING';
    #model = train_svm_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
    model_bicycling = train_rf_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
    #print modelbi
    # In[40]:
    
    
    #uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/11B5EC4D-4133-4289-B475-4E737182A406';
    #(X2,Y2,M2,timestamps2,feature_names2,label_names2) = read_user_data(uuid);
    
    # All the user data files should have the exact same columns. We can validate it:
    #validate_column_names_are_consistent(feature_names,feature_names2);
    #validate_column_names_are_consistent(label_names,label_names2);
    
    
    # In[41]:
    
    
    #(a,b,p,y_prob_bicylcling,y_label_bicycling) = test_svm_model(X,Y,M,timestamps,feat_sensor_names,label_names,model);
    (a,b,p,y_prob_bicylcling,y_label_bicycling) = test_rf_model(X,Y,M,timestamps,feat_sensor_names,label_names,model_bicycling);    
   #%%
      #%%  #%%#%%#%%#%%#%%
    #sensors_to_use = ['Acc','WAcc','Gyro','Magnet','Compass','Aud'];
    #sensors_to_use = ['Acc','Gyro','Magnet'];
    sensors_to_use = ['Acc','Gyro','Magnet','loc'];
    #sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud'];
    #sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud','WAcc','Compass'];
    target_label = 'LYING_DOWN';
    #model = train_svm_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
    model_lying = train_rf_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
    #print modelbi
    # In[40]:
    
    
    #uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/11B5EC4D-4133-4289-B475-4E737182A406';
    #(X2,Y2,M2,timestamps2,feature_names2,label_names2) = read_user_data(uuid);
    
    # All the user data files should have the exact same columns. We can validate it:
    #validate_column_names_are_consistent(feature_names,feature_names2);
    #validate_column_names_are_consistent(label_names,label_names2);
    
    
    # In[41]:
    
    
    #(a,b,p,y_prob_bicylcling,y_label_bicycling) = test_svm_model(X,Y,M,timestamps,feat_sensor_names,label_names,model);
    (a,b,p,y_prob_lying,y_label_lying) = test_rf_model(X,Y,M,timestamps,feat_sensor_names,label_names,model_lying);    
   #%%
      #%%  #%%#%%#%%#%%#%%
    #sensors_to_use = ['Acc','WAcc','Gyro','Magnet','Compass','Aud'];
    #sensors_to_use = ['Acc','Gyro','Magnet'];
    sensors_to_use = ['Acc','Gyro','Magnet','loc'];
    #sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud'];
    #sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud','WAcc','Compass'];
    target_label = 'SITTING';
    #model = train_svm_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
    model_sitting = train_rf_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
    #print modelbi
    # In[40]:
    
    
    #uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/11B5EC4D-4133-4289-B475-4E737182A406';
    #(X2,Y2,M2,timestamps2,feature_names2,label_names2) = read_user_data(uuid);
    
    # All the user data files should have the exact same columns. We can validate it:
    #validate_column_names_are_consistent(feature_names,feature_names2);
    #validate_column_names_are_consistent(label_names,label_names2);
    
    
    # In[41]:
    
    
    #(a,b,p,y_prob_bicylcling,y_label_bicycling) = test_svm_model(X,Y,M,timestamps,feat_sensor_names,label_names,model);
    (a,b,p,y_prob_sitting,y_label_sitting) = test_rf_model(X,Y,M,timestamps,feat_sensor_names,label_names,model_sitting);    
   #%%
      #%%  #%%#%%#%%#%%#%%
    #sensors_to_use = ['Acc','WAcc','Gyro','Magnet','Compass','Aud'];
    #sensors_to_use = ['Acc','Gyro','Magnet'];
    sensors_to_use = ['Acc','Gyro','Magnet','loc'];
    #sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud'];
    #sensors_to_use = ['Acc','Gyro','Magnet','loc','Aud','WAcc','Compass'];
    target_label = 'OR_standing';
    #model = train_svm_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
    model_standing = train_rf_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
    #print modelbi
    # In[40]:
    
    
    #uuid = '/home/abraham/Documents/Rotation_two/Existing_python_implementation/Activity_Classification_with_UCSDdataset/11B5EC4D-4133-4289-B475-4E737182A406';
    #(X2,Y2,M2,timestamps2,feature_names2,label_names2) = read_user_data(uuid);
    
    # All the user data files should have the exact same columns. We can validate it:
    #validate_column_names_are_consistent(feature_names,feature_names2);
    #validate_column_names_are_consistent(label_names,label_names2);
    
    
    # In[41]:
    
    
    #(a,b,p,y_prob_bicylcling,y_label_bicycling) = test_svm_model(X,Y,M,timestamps,feat_sensor_names,label_names,model);
    (a,b,p,y_prob_stand,y_label_stand) = test_rf_model(X,Y,M,timestamps,feat_sensor_names,label_names,model_standing);    
   
    # In[41]:
    
    #print ("The length of indoor lochome and meeting are %s %s %s "% (len(y_prob_indoors), len(y_prob_lochome), len(y_prob_meeting)))                                                                                                                                                                                                                                                                                                                                                                                                
    
    length_min_Act_Inactive=min(len(y_prob_active),len(y_prob_InActive))
    length_min_USER_active=min(len(y_prob_active),len(y_prop_Transport),len(y_prob_userprop),len(y_prob_bicylcling),len(y_prob_walking),len(y_prob_Driver),len(y_prob_passenger))
    length_min_second_level=min(len(y_prob_active),len(y_prop_Transport),len(y_prob_userprop),len(y_prob_stand),len(y_prob_sitting),len(y_prob_InActive),len(y_prob_lying))
    
    y_prob_user_Active=np.zeros(shape=(length_min_Act_Inactive,1))
    y_prob_user_Inactive=np.zeros(shape=(length_min_Act_Inactive,1))   
    y_prob_user_passenger=np.zeros(shape=(length_min_Act_Inactive,1))
    y_prob_user_driver=np.zeros(shape=(length_min_Act_Inactive,1))
    y_prob_user_walking=np.zeros(shape=(length_min_Act_Inactive,1))
    y_prob_user_bicyling=np.zeros(shape=(length_min_Act_Inactive,1))
    
    y_prob_user_lying=np.zeros(shape=(length_min_second_level,1))
    y_prob_user_sitting=np.zeros(shape=(length_min_second_level,1))
    y_prob_user_standing=np.zeros(shape=(length_min_second_level,1))
    y_prob_user_Transport=np.zeros(shape=(length_min_second_level,1))
    y_prob_user_userprop=np.zeros(shape=(length_min_second_level,1))

    
    y_comp_user_Active=np.zeros(shape=(length_min_Act_Inactive,1))
    y_comp_user_InActive=np.zeros(shape=(length_min_Act_Inactive,1))
    
    y_comp_user_passenger=np.zeros(shape=(length_min_Act_Inactive,1))
    y_comp_user_driver=np.zeros(shape=(length_min_Act_Inactive,1))
    y_comp_user_walking=np.zeros(shape=(length_min_Act_Inactive,1))
    y_comp_user_bicyling=np.zeros(shape=(length_min_Act_Inactive,1))
    
    y_comp_user_lying=np.zeros(shape=(length_min_second_level,1))
    y_comp_user_sitting=np.zeros(shape=(length_min_second_level,1))
    y_comp_user_standing=np.zeros(shape=(length_min_second_level,1))
    y_comp_user_Transport=np.zeros(shape=(length_min_second_level,1))
    y_comp_user_userprop=np.zeros(shape=(length_min_second_level,1))
    
    y_pred_user_Active=np.zeros(shape=(length_min_Act_Inactive,1))
    y_pred_user_InActive=np.zeros(shape=(length_min_Act_Inactive,1))
    
    y_pred_user_passenger=np.zeros(shape=(length_min_Act_Inactive,1))
    y_pred_user_driver=np.zeros(shape=(length_min_Act_Inactive,1))
    y_pred_user_walking=np.zeros(shape=(length_min_Act_Inactive,1))
    y_pred_user_bicycling=np.zeros(shape=(length_min_Act_Inactive,1))
       
    y_pred_user_lying=np.zeros(shape=(length_min_second_level,1))
    y_pred_user_sitting=np.zeros(shape=(length_min_second_level,1))
    y_pred_user_standing=np.zeros(shape=(length_min_second_level,1))
    y_pred_user_transport=np.zeros(shape=(length_min_second_level,1))
    y_pred_user_userprop=np.zeros(shape=(length_min_second_level,1))
    #sum_test=0       
    #change the existing label from boealean to number
    for i in range(0,length_min_Act_Inactive):
        if (y_label_Active[i]==True):
            y_pred_user_Active[i] = 1
            
        else:
            y_pred_user_Active[i] = 0 
            
        if (y_label_InActive [i]==True):
            y_pred_user_InActive[i] = 1
           # sum_test=y_pred_user_InActive[i] + sum_test
        else:
            y_pred_user_InActive[i] = 0 
   
    #change the existing label from boealean to number
    for i in range(0,length_min_USER_active):
        if (y_label_passenger[i]==True):
            y_pred_user_passenger[i] = 1
            
        else:
            y_pred_user_passenger[i] = 0 
            
        if (y_label_Driver [i]==True):
            y_pred_user_driver[i] = 1
          
        else:
            y_pred_user_driver[i] = 0   
            
        if (y_label_walking [i]==True):
            y_pred_user_walking[i] = 1
            
        else:
            y_pred_user_walking[i] = 0 
            
        if (y_label_bicycling [i]==True):
            y_pred_user_bicycling[i] = 1
          
        else:
            y_pred_user_bicycling[i] = 0           
    

        #change the existing label from boealean to number
    for i in range(0,length_min_second_level):
        if (y_label_Transport[i]==True):
            y_pred_user_transport[i] = 1
        else:
            y_pred_user_transport[i] = 0 
            
        if (y_label_userprop [i]==True):
            y_pred_user_userprop[i] = 1
        else:
            y_pred_user_userprop[i] = 0   
            
        if (y_label_lying [i]==True):
            y_pred_user_lying[i] = 1
        else:
            y_pred_user_lying[i] = 0 
            
        if (y_label_sitting [i]==True):
            y_pred_user_sitting[i] = 1
        else:
            y_pred_user_sitting[i] = 0  
        
        if (y_label_stand [i]==True):
            y_pred_user_standing[i] = 1
        else:
            y_pred_user_standing[i] = 0     
         
    #finding probabilities of user active inactive location
    for i in range (0,length_min_Act_Inactive):
        y_prob_user_Active[i]=   y_prob_active [i,1]
        y_prob_user_Inactive[i] =   y_prob_InActive [i,1]
     
     #finding probabilities of user active location
    for i in range (0,length_min_USER_active):
        y_prob_user_passenger[i]= y_prob_passenger[i,1] * y_prop_Transport[i,1] * y_prob_active[i,1]
        y_prob_user_driver[i]=  y_prob_Driver [i,1] * y_prop_Transport[i,1] * y_prob_active[i,1]
        y_prob_user_walking[i]=   y_prob_walking [i,1] * y_prob_userprop[i,1] * y_prob_active[i,1]
        y_prob_user_bicyling[i] =   y_prob_bicylcling [i,1] * y_prob_userprop[i,1] * y_prob_active[i,1]
      
        
    #finding probabilities of user mobility
    for i in range (0,length_min_second_level):
        y_prob_user_Transport[i]= y_prob_active[i,1] * y_prop_Transport[i,1]
        y_prob_user_userprop[i]= y_prob_active[i,1] * y_prob_userprop [i,1]
        y_prob_user_lying[i]= y_prob_InActive[i,1] * y_prob_lying [i,1]   
        y_prob_user_sitting[i]= y_prob_InActive[i,1] * y_prob_sitting[i,1]
        y_prob_user_standing[i] = y_prob_InActive[i,1] * y_prob_stand[i,1]
       
    
            #%%
            
    #comparingphone probabilities phone location
    for i in range (0, length_min_Act_Inactive):
        max_value=max(y_prob_user_Active[i],y_prob_user_Inactive[i])
      
            #y_prob_indoor_gym[i] =0
        #elif (y_prob_user_Active[i] == max_value):
            
        if (y_prob_user_Inactive[i] == max_value):
        #else:
            y_comp_user_Active[i]=0
            y_comp_user_InActive[i]=1 
        if (y_prob_user_Active[i] == max_value):
        #else:
            y_comp_user_Active[i]= 1
            y_comp_user_InActive[i]= 0
        
       #%%              #%%
       #comparingphone probabilities user  location
    for i in range (0, length_min_USER_active):
        max_value=max(y_prob_user_passenger[i],y_prob_user_driver[i],y_prob_user_walking[i],y_prob_user_bicyling[i])
        if (y_prob_user_bicyling[i] == max_value):
        #else:
            y_comp_user_passenger[i] = 0
            y_comp_user_driver[i] = 0
            y_comp_user_walking[i]=0
            y_comp_user_bicyling[i]=1 
            
        elif (y_prob_user_passenger[i] == max_value):
            y_comp_user_passenger[i] = 1
            y_comp_user_driver[i] = 0
            y_comp_user_walking[i]= 0
            y_comp_user_bicyling[i]= 0
            
           # y_comp_indoor_gym[i] =0
        elif (y_prob_user_driver[i] == max_value):
        #elif (y_prob_user_driver[i] == max_value):
            y_comp_user_passenger[i] = 0
            y_comp_user_driver[i] = 1
            y_comp_user_walking[i]=0
            y_comp_user_bicyling[i]=0
            #y_prob_indoor_gym[i] =0
        #elif (y_prob_user_walking[i] == max_value):
        #x`if (y_prob_user_walking[i] == max_value):
        else:
            y_comp_user_passenger[i] = 0
            y_comp_user_driver[i] = 0
            y_comp_user_walking[i]= 1
            y_comp_user_bicyling[i]= 0
            
      
    #%%
        #comparing the user mobility probabilities
    for i in range (0, length_min_second_level):
       
    
        max_value=max(y_prob_user_Transport[i],y_prob_user_userprop[i],y_prob_user_lying[i],y_prob_user_sitting[i],y_prob_user_standing[i])
              #y_prob_indoor_gym[i] =0
        if (y_prob_user_lying[i] == max_value):
            y_comp_user_Transport[i] = 0
            y_comp_user_userprop[i] = 0
            y_comp_user_lying[i] = 1
            y_comp_user_sitting[i] = 0
            y_comp_user_standing[i] = 0
          
        elif (y_prob_user_Transport[i] == max_value):
            y_comp_user_Transport[i] = 1
            y_comp_user_userprop[i] = 0
            y_comp_user_lying[i] = 0
            y_comp_user_sitting[i] = 0
            y_comp_user_standing[i] = 0
          
           # y_comp_indoor_gym[i] =0
        elif (y_prob_user_userprop[i] == max_value):
            y_comp_user_Transport[i] = 0
            y_comp_user_userprop[i] = 1
            y_comp_user_lying[i] = 0
            y_comp_user_sitting[i] = 0
            y_comp_user_standing[i] = 0
           
      
    
            #y_comp_indoor_gym[i] =0
        elif (y_prob_user_sitting[i] == max_value):
            y_comp_user_Transport[i] = 0
            y_comp_user_userprop[i] = 0
            y_comp_user_lying[i] = 0
            y_comp_user_sitting[i] = 1
            y_comp_user_standing[i] = 0
            
    
        elif (y_prob_user_standing[i] == max_value):
            y_comp_user_Transport[i] = 0
            y_comp_user_userprop[i] = 0
            y_comp_user_lying[i] = 0
            y_comp_user_sitting[i] = 0
            y_comp_user_standing[i] = 1
       
    
       
   #%%
    y_pred_act_inact=np.zeros(shape=(length_min_Act_Inactive,1))
    y_orig_act_inact=np.zeros(shape=(length_min_Act_Inactive,1))
    y_pred_user=np.zeros(shape=(length_min_USER_active,1))
    y_orig_user=np.zeros(shape=(length_min_USER_active,1))
    y_pred_second_level= np.zeros(shape=(length_min_second_level,1))
    y_orig_second_level= np.zeros(shape=(length_min_second_level,1))
   # F1_score_act_inactive=np.zeros(shape=(cont_num,UUID_length))
    
     #%%   
    #pr
    #find the accuracy and balanced accuracy for phone location
    for i in range(0,cont_num):
        if i==0:
        #else:
            y_pred_act_inact=y_comp_user_Active
            y_orig_act_inact=y_pred_user_Active
            string_value="Active  "
        #if i==1:
        else:
            y_pred_act_inact=y_comp_user_InActive
            y_orig_act_inact=y_pred_user_InActive
            string_value="Inactive  "
            
        tp = np.sum(np.logical_and(y_pred_act_inact,y_orig_act_inact));
        tn = np.sum(np.logical_and(np.logical_not(y_pred_act_inact),np.logical_not(y_orig_act_inact)));
        fp = np.sum(np.logical_and(y_pred_act_inact,np.logical_not(y_orig_act_inact)));
        fn = np.sum(np.logical_and(np.logical_not(y_pred_act_inact),y_orig_act_inact));
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
            F1_score_act_inactive[i] [input_value]= 2 * (sensitivity * precision) / (sensitivity + precision)
      
        # Balanced accuracy is a more fair replacement for the naive accuracy:
        balanced_accuracy_phone[i][input_value]=(sensitivity + specificity) / 2.;
        accuracy_phone[i][input_value]=np.mean(y_pred_act_inact==y_orig_act_inact);
       # print('the accuracy, balanced accuracy, F1 score of %s is : %f %f %f '% (string_value, accuracy_phone[i],balanced_accuracy_phone[i],F1_score_act_inactive[i]))
        #print('the F1 score of %s is :  %f '% (string_value, F1_score_act_inactive[i][input_value]))
#%%
           #%%   
    #pr
   
    #find the accuracy and balanced accuracy for user location
    for i in range(0,count_user_act):
        if i==0:
            y_pred_user=y_comp_user_passenger
            y_orig_user=y_pred_user_passenger
            string_value="passenger "
        if i==1:
        #if i==1:
            y_pred_user=y_comp_user_driver
            y_orig_user=y_pred_user_driver
            string_value="driver  "
        if i==2:
        #else:
            y_pred_user=y_comp_user_walking
            y_orig_user=y_pred_user_walking
            string_value="walking"
        if i==3:
        #else:
            y_pred_user=y_comp_user_bicyling
            y_orig_user=y_pred_user_bicycling
            string_value="bicycling"
            
        tp = np.sum(np.logical_and(y_pred_user,y_orig_user));
        tn = np.sum(np.logical_and(np.logical_not(y_pred_user),np.logical_not(y_orig_user)));
        fp = np.sum(np.logical_and(y_pred_user,np.logical_not(y_orig_user)));
        fn = np.sum(np.logical_and(np.logical_not(y_pred_user),y_orig_user));
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
            F1_score_user[i] [input_value]= 2 * (sensitivity * precision) / (sensitivity + precision)
      
        # Balanced accuracy is a more fair replacement for the naive accuracy:
        #balanced_accuracy_phone[i][input_value]=(sensitivity + specificity) / 2.;
        #accuracy_phone[i][input_value]=np.mean(y_pred_user==y_orig_user);
       # print('the accuracy, balanced accuracy, F1 score of %s is : %f %f %f '% (string_value, accuracy_phone[i],balanced_accuracy_phone[i],F1_score_user[i]))
       # print('the F1 score of %s is :  %f '% (string_value, F1_score_user[i][input_value]))
#%% y_comp_user_lying=np.zeros(shape=(length_min_second_level,1))
   
      
         #find the accuracy and balanced accuracy for phone location
    for i in range(0,count_second_level):
        if i==0:
            y_pred_second_level=y_comp_user_Transport
            y_orig_second_level=y_pred_user_transport
            string_value="Transport "
        if i==1:
        #if i==1:
            y_pred_second_level=y_comp_user_userprop
            y_orig_second_level=y_pred_user_userprop
            string_value="userprop "
        if i==2:
        #else:
            y_pred_second_level=y_comp_user_lying
            y_orig_second_level=y_pred_user_lying
            string_value="lying"
        if i==3:
        #else:
            y_pred_second_level=y_comp_user_sitting
            y_orig_second_level=y_pred_user_sitting
            string_value="sitting"
        if i==4:
        #else:
            y_pred_second_level=y_comp_user_standing
            y_orig_second_level=y_pred_user_standing
            string_value="standing"   
        tp = np.sum(np.logical_and(y_pred_second_level,y_orig_second_level));
        tn = np.sum(np.logical_and(np.logical_not(y_pred_second_level),np.logical_not(y_orig_second_level)));
        fp = np.sum(np.logical_and(y_pred_second_level,np.logical_not(y_orig_second_level)));
        fn = np.sum(np.logical_and(np.logical_not(y_pred_second_level),y_orig_second_level));
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
            F1_score_second_level[i] [input_value]= 2 * (sensitivity * precision) / (sensitivity + precision)
      
        # Balanced accuracy is a more fair replacement for the naive accuracy:
        #balanced_accuracy_phone[i][input_value]=(sensitivity + specificity) / 2.;
       # accuracy_phone[i][input_value]=np.mean(y_pred_second_level==y_orig_second_level);
       # print('the accuracy, balanced accuracy, F1 score of %s is : %f %f %f '% (string_value, accuracy_phone[i],balanced_accuracy_phone[i],F1_score_second_level[i]))
        #print('the F1 score of %s is :  %f '% (string_value, F1_score_second_level[i][input_value]))

        #%%
sum_Active=0
sum_InActive=0
sum_Active_div = 0
sum_InActive_div = 0
for j in range (0,cont_num):
    for i in range(input_init,UUID_length):   
        if j==0:
        #else:
            string_value="Active  "
            sum_Active=sum_Active+ F1_score_act_inactive[j][i]
            if F1_score_act_inactive[j][i] != 0:
                sum_Active_div = sum_Active_div+1
        #if j==1:
        else:
            string_value="InActive"
            sum_InActive=sum_InActive+ F1_score_act_inactive[j][i]
            if F1_score_act_inactive[j][i] != 0:
                sum_InActive_div = sum_InActive_div + 1
        #print('the F1 score of %s is : %f'% (string_value, F1_score_act_inactive[j][i]));

average_Active=sum_Active/sum_Active_div;
average_InActive= sum_InActive/sum_InActive_div;

print('the average F1 score of %s %s is :  %f %f'% ('Active', 'InActive', average_Active, average_InActive));
print('the sum F1 score of %s %s is :  %f %f'% ('Active', 'InActive', sum_Active, sum_InActive));
                
    #print('The balanced accuracy are %2.f' % balanced_accuracy_phone[0])
    #print('accuracy is:       %.2f' % accuracy_phone[1])
    #%%
sum_passenger=0;
sum_driver=0;
sum_walking=0
sum_bicycling=0
sum_passenger_div = 0
sum_driver_div = 0
sum_walking_div = 0
sum_bicycling_div = 0
for j in range (0,count_user_act):
    for i in range(input_init,UUID_length):   
        if j==0:
            string_value="passenger "
            sum_passenger=sum_passenger+ F1_score_user[j][i]
            if F1_score_user[j][i] != 0:
                sum_passenger_div = sum_passenger_div + 1
        if j==1:
        #if j==1:
            string_value="driver  "
            sum_driver=sum_driver+ F1_score_user[j][i]
            if F1_score_user[j][i] != 0:
                sum_driver_div = sum_driver_div+1
        if j==2:
        #else:
            string_value="walking    "
            sum_walking=sum_walking+ F1_score_user[j][i]
            if F1_score_user[j][i] != 0:
                sum_walking_div = sum_walking_div+1
        if j==3:
        #else:
            string_value="bicylcing"
            sum_bicycling=sum_bicycling+ F1_score_user[j][i]
            if F1_score_user[j][i] != 0:
                sum_bicycling_div = sum_bicycling_div + 1
        #print('the F1 score of %s is : %f'% (string_value, F1_score_user[j][i]));

average_passenger=sum_passenger/sum_passenger_div;
average_driver=sum_driver/sum_driver_div;
average_walking=sum_walking/sum_walking_div;
average_bicycling= sum_bicycling/sum_bicycling_div;

print('the average F1 score of %s %s %s %s is : %f %f %f %f'% ('passenger','driver', 'walking', 'bicyling', average_passenger,average_driver,average_walking, average_bicycling));
print('the sum F1 score of %s %s %s %s is : %f %f %f %f'% ('pocket','table', 'hand', 'BAG', sum_passenger,sum_driver,sum_walking, sum_bicycling));
                
    #print('The balanced accuracy are %2.f' % balanced_accuracy_phone[0])
    #pr
#%%
sum_Transport=0;
sum_userprop=0;
sum_lying=0
sum_sitting=0
sum_standing=0
sum_Transport_div = 0
sum_userprop_div = 0
sum_lying_div = 0
sum_sitting_div = 0
sum_standing_div =0
for j in range (0,count_second_level):
    for i in range(input_init,UUID_length):   
        if j==0:
            string_value="transport "
            sum_Transport=sum_Transport+ F1_score_second_level[j][i]
            if F1_score_second_level[j][i] != 0:
                sum_Transport_div = sum_Transport_div + 1
        if j==1:
        #if j==1:
            string_value="userprop  "
            sum_userprop=sum_userprop+ F1_score_second_level[j][i]
            if F1_score_second_level[j][i] != 0:
                sum_userprop_div = sum_userprop_div+1
        if j==2:
        #else:
            string_value="lying"
            sum_lying=sum_lying+ F1_score_second_level[j][i]
            if F1_score_second_level[j][i] != 0:
                sum_lying_div = sum_lying_div+1
        if j==3:
        #else:
            string_value="sitting"
            sum_sitting=sum_sitting+ F1_score_second_level[j][i]
            if F1_score_second_level[j][i] != 0:
                sum_sitting_div = sum_sitting_div + 1
        if j==4:
        #else:
            string_value="standing"
            sum_standing=sum_standing+ F1_score_second_level[j][i]
            if F1_score_second_level[j][i] != 0:
                sum_standing_div = sum_standing_div + 1        
       # print('the F1 score of %s is : %f'% (string_value, F1_score_second_level[j][i]));

average_transport=sum_Transport/sum_Transport_div;
average_userprop=sum_userprop/sum_userprop_div;
average_lying=sum_lying/sum_lying_div;
average_sitting= sum_sitting/sum_sitting_div;
average_standing= sum_standing/sum_standing_div
print('the average F1 score of %s %s %s %s %s is : %f %f %f %f %f'% ('transport','userprop', 'lying', 'sitting','standing', average_transport,average_userprop,average_lying, average_sitting,average_standing));
print('the sum F1 score of %s %s %s %s %s is : %f %f %f %f %f'% ('transport','userprop', 'lying', 'sitting', 'standing',sum_Transport,sum_userprop,sum_lying, sum_sitting,sum_standing));
