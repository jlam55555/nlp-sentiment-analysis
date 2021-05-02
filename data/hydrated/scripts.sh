# creating all.csv
cat ../labels/*.csv all.csv

# converting jsonl to json
echo '[' >all_unlabeled.json
cat all.jsonl\
	|jq '{id:.id_str,text:.full_text}' -c -M\
	|sed 's/$/,/g'\
	|sed '$ s/,$/]/g' >>all_unlabeled.json

# then run combine.py
# to combine with labels
