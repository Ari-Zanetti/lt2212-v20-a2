# LT2212 V20 Assignment 2

Part 1 - creating the feature table

The function "extract_features(samples)" loops through the given samples and selects the words to use as features.
First, it lower-cases each word, then it excludes the numbers and removes punctuation marks from the remaining strings.
When the feature table is created, it removes unfrequent words, in particular words that do not appear more than 5 times in any of the documents, and words that appear in more than half of the documents (likely to be stop-words).
The number of selected features after the filter is: 6820.


Part 3 - classify and evaluate 

--model_id = 1 Creates a Support Vector Classifier
--model_id = 2 Creates a Decision Tree Classifier


Part 4 - try and discuss

1.  SVC:

	Accuracy: 0.6153846153846154
	Precision: 0.7312835620468641
	Recall: 0.6153846153846154
	F-measure: 0.6295393242562133


	Decision tree:

	Accuracy: 0.6289124668435013
	Precision: 0.6345327387873417
	Recall: 0.6289124668435013
	F-measure: 0.630396825430384
	
	
2.  50% (3000 features)

	SVC:

	Accuracy: 0.6647214854111406
	Precision: 0.7450556008221445
	Recall: 0.6647214854111406
	F-measure: 0.6742866308812862


	Decision tree:

	Accuracy: 0.2339522546419098
	Precision: 0.23712542693859126
	Recall: 0.2339522546419098
	F-measure: 0.23484380138558344
	
	
	25% (1500 features)

	SVC:

	Accuracy: 0.6814323607427055
	Precision: 0.7434391482719998
	Recall: 0.6814323607427055
	F-measure: 0.6884433865720662


	Decision tree:

	Accuracy: 0.24350132625994694
	Precision: 0.2427450709627398
	Recall: 0.24350132625994694
	F-measure: 0.24269421907822789
		
	
	10% (600 features)

	SVC:

	Accuracy: 0.6787798408488064
	Precision: 0.7230790830118445
	Recall: 0.6787798408488064
	F-measure: 0.6830548214650765


	Decision tree:

	Accuracy: 0.2509283819628647
	Precision: 0.25457217698524215
	Recall: 0.2509283819628647
	F-measure: 0.25203599577295216
		
	
	5% (300 features)

	SVC:

	Accuracy: 0.6429708222811671
	Precision: 0.6861023190984168
	Recall: 0.6429708222811671
	F-measure: 0.6466920622707599


	Decision tree:

	Accuracy: 0.2572944297082228
	Precision: 0.26141784913178323
	Recall: 0.2572944297082228
	F-measure: 0.2584856114475635
	
	
What we can see from the results is that the Support Vector Classifier has similar values of precision and recall with the reduced features, around 65%/70% on average. 
It seems to even work a bit better when the features are 25% or even 10% of the original number, even if it could just be chance.
The decision tree classifier, on the contrary, has comparable values of precision and recall to SVC when used on the complete training set (62% and 63%), but they are drastically decreased to 23%/24% when used on the reduced features, already with 50% of the data and reducing even more does not make a big difference.


Part Bonus - another dimensionality reduction

In part 2, I used Principal component analysis for dimensionality reduction, so in part bonus I decided to use Truncated SVD.
To use the alternative dimensionality reduction, the script has to be called with the additional parameter -r (--alternative_reduction).

Results:

	50% (3000 features)

	SVC:

	Accuracy: 0.6633952254641909
	Precision: 0.7439853719254946
	Recall: 0.6633952254641909
	F-measure: 0.6728091934627611


	Decision tree:

	Accuracy: 0.23474801061007958
	Precision: 0.23694141234071014
	Recall: 0.23474801061007958
	F-measure: 0.23501969441285195
	
	
	25% (1500 features)

	SVC:

	Accuracy: 0.6816976127320955
	Precision: 0.7436190781941717
	Recall: 0.6816976127320955
	F-measure: 0.6888051423045148
	

	Decision tree:
	
	Accuracy: 0.2440318302387268
	Precision: 0.2466419643714602
	Recall: 0.2440318302387268
	F-measure: 0.24437638774067905


	10% (600 features)

	SVC:

	Accuracy: 0.6779840848806366
	Precision: 0.7233266660870354
	Recall: 0.6779840848806366
	F-measure: 0.681847270332927
	

	Decision tree:

	Accuracy: 0.24297082228116712
	Precision: 0.24613071351205032
	Recall: 0.24297082228116712
	F-measure: 0.24385754232075632

	
	5% (300 features)

	SVC:

	Accuracy: 0.6456233421750663
	Precision: 0.6864301485343208
	Recall: 0.6456233421750663
	F-measure: 0.6481380846632142
	

	Decision tree:
	
	Accuracy: 0.259946949602122
	Precision: 0.26156888670090034
	Recall: 0.259946949602122
	F-measure: 0.2599534294092134

	
With the alternative dimensionality reduction, we can confirm our observations: the Support Vector Classifier maintains values of precision and recall between 65%/70% on average, while the Decision Tree Classifier doesn't work well with any kind of dimensionality reduction, having values of precision and recall around 24%/25%.


	

  
	
	