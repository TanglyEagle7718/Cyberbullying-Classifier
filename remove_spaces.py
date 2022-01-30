import csv
input = open('new_cyberbullying_data.csv', 'r')
output = open('blankless_cyberbullying_data.csv', 'wt')
writer = csv.writer(output)
for row in csv.reader(input):
    if row[0]!="`1234567890-=`1234567890-=":
        writer.writerow(row)
input.close()
output.close()