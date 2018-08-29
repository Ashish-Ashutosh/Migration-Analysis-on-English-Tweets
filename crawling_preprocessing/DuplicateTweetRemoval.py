# Defines a list - It stores all unique tweets

import csv
tweetChecklist = [];


list = []

fh = open('/home/stark/dataS/Favour.csv', 'r')

for line in fh:
    list.append(line.strip().split('\n'))

# All your tweets. I represent them as a list to test the code



# Goes over all "tweets"

for current_tweet in list:
        # If tweet doesn't exist in the list
        if current_tweet not in tweetChecklist:
            tweetChecklist.append(current_tweet);
            print(current_tweet)
            # Do what you want with this tweet, it won't appear two times...

# Print ["Hello", "HelloFoo", "HelloBar", "hello", "Bye"]
# Note that the second Hello doesn't show up - It's what you want
# However, it's case sensitive.

print(tweetChecklist);
# Clear the list


#res = [x, y, z, ....]

csvfile = '/home/stark/dataS/FavourUni.csv'

#Assuming res is a flat list
#with open(csvfile, "w") as output:
 #   writer = csv.writer(output, lineterminator='\n')
  #  for val in res:
   #     writer.writerow([val])

#Assuming res is a list of lists


with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(tweetChecklist)
