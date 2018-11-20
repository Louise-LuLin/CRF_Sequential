package gLearner;

import edu.umass.cs.mallet.grmm.types.*;
import gLearner.GraphLearner;
import gLearner.SeqAnalyzer;
import gLearner.String4Learning;

import java.util.ArrayList;
import java.util.Random;

/**
 *  This is a class putting node and edge features into table factors.
 */

public class CRF {

    //Sequence analyzer to access all data and vocabulary
    SeqAnalyzer m_seq;

    public CRF( SeqAnalyzer seq){
        m_seq = seq;
    }

    public SeqAnalyzer getSeq(){ return this.m_seq; }

    private double[] calcAcc(ArrayList<int[]> true_labels, ArrayList<ArrayList<Integer>> pred_labels ){
        double[] accs = new double[3]; //all_acc, phrase_acc, out_acc
        if(true_labels.size() != pred_labels.size()) {
            System.err.format("[Err]Prediction has different length than ground truth.\n");
            return accs;
        }

        double[] counts = new double[6]; //all_correct, phrase_correct, out_correct, all_count, phrase_count, out_count
        for(int i = 0; i < true_labels.size(); i++){
            boolean correctFlag = false;
            int label_idx_true, label_idx_pred;
            String label_true;
            String label_pred;
            for(int j = 0; j < true_labels.get(i).length; j++){
                label_idx_true = true_labels.get(i)[j];
                label_true = m_seq.getLabelNames().get(label_idx_true);
                label_idx_pred = pred_labels.get(i).get(j);
                label_pred = m_seq.getLabelNames().get(label_idx_pred);
                //If the character is a beginning-of-phrase.
                if(label_true.startsWith("b")){
                    counts[3+1] += 1;
                    if(label_idx_true == label_idx_pred){
                        if(correctFlag)
                            counts[1] += 1;
                        correctFlag = true;
                    } else {
                        if(correctFlag){
                            if(!label_true.substring(2).equals(
                                    m_seq.getLabelNames().get(pred_labels.get(i).get(j-1)).substring(2))) //special case
                                counts[1] += 1;
                        }
                        correctFlag = false;
                    }
                }//If the character is an inside-of-phrase.
                else if(label_true.startsWith("i")){
                    if(label_idx_pred != label_idx_true)
                        correctFlag = false;
                }//If the character is an out-of-phrase.
                else if(label_true.startsWith("o")){
                    counts[3+2] += 1;
                    if(label_idx_pred == label_idx_true){
                        counts[2] += 1;
                        if(correctFlag){
                            counts[1] += 1;
                            correctFlag = false;
                        }

                    } else {
                        if(correctFlag){
                            if(!label_pred.substring(2).equals(
                                    m_seq.getLabelNames().get(pred_labels.get(i).get(j-1)).substring(2))) //special case
                                counts[1] += 1;
                            correctFlag = false;
                        }

                    }

                }
            }
            //For the case where the phrase is at the end of a string.
            if(correctFlag)
                counts[1] += 1;
        }
        counts[0] = counts[1] + counts[2];
        counts[3] = counts[3+1] + counts[3+2];

        for(int i = 0; i < 3; i++)
            accs[i] = counts[i]/counts[i+3];

        return accs;
    }

    public void crossValidation(int k, String prefix){
    	

        double[][] acc = new double[k][3];

        int[] masks = new int[m_seq.getStrings().size()];
        Random rand = new Random();
        for(int i = 0; i < masks.length; i++)
            masks[i] = rand.nextInt(k);

        System.out.format("[Info]Start RANDOM cross validation...\n");        
        
        //use this loop to iterate all the folders, set train and test
        for(int i = 0; i < k; i++){
        	ArrayList<int[]> train_label = new ArrayList<int[]>();
            ArrayList<String4Learning> training_data = new ArrayList<String4Learning>();

        	ArrayList<int[]> test_label = new ArrayList<int[]>();
            ArrayList<Sequence> testing_seq = new ArrayList<Sequence>();
        	
            for(int j = 0; j < masks.length; j++){
                if(masks[j] == i){
                    test_label.add(m_seq.getLabels().get(j));
                    testing_seq.add(m_seq.getSequences().get(j));
                }else{
                    train_label.add(m_seq.getLabels().get(j));
                    training_data.add(m_seq.getStr4Learning(m_seq.getSequences().get(j), "train"));
                }
            }

            System.out.format("==========\n[Info]Fold No. %d: train size = %d, test size = %d...\n",
                    i, training_data.size(), testing_seq.size());
            
            // Build up a graph learner and train it using training data.
            GraphLearner m_graphLearner = new GraphLearner(training_data);

            // Train
            long start = System.currentTimeMillis();
            ArrayList<ArrayList<Integer>> trainPrediction = m_graphLearner.doTraining(1);
            double acc_cur = calcAcc(train_label, trainPrediction)[0];
            System.out.format("cur train acc: %f\n", acc_cur);

            m_graphLearner.SaveWeights(String.format("%s/weights.txt", prefix));

            // Apply the trained model to the test set.
            ArrayList<ArrayList<Integer>> testPrediction = new ArrayList<>();
            ArrayList<Integer> pred_tmp;
            int j=0;
            for(Sequence seq : testing_seq) {
                System.out.format("-- test sample %d\n", j++);
                FactorGraph testGraph = m_graphLearner.buildFactorGraph_test(m_seq.getStr4Learning(seq, "test"));
                pred_tmp = m_graphLearner.doTesting(testGraph);
                testPrediction.add(pred_tmp);
                System.out.format("[debug]predicted lable: %d, %d, %d, %d, %d\n",
                        pred_tmp.get(0), pred_tmp.get(1), pred_tmp.get(2), pred_tmp.get(3), pred_tmp.get(4));
            }
            acc[i] = calcAcc(test_label, testPrediction);

            System.out.format("[Stat]Train/test finished in %.2f seconds: acc_all = %.2f, acc_phrase = %.2f, acc_out = %.2f\n",
                    (System.currentTimeMillis()-start)/1000.0, acc[i][0], acc[i][1], acc[i][2]);
        }

        double[] mean = new double[3];
        double[] var = new double[3];
        for(int i = 0; i < k; i++){
            for(int j = 0; j < 3; j++)
                mean[j] += acc[i][j];
        }
        for(int j = 0; j < 3; j++)
            mean[j] /= k;

        for(int i = 0; i < k; i++){
            for(int j = 0; j < 3; j++){
                var[j] += (acc[i][j] - mean[j]) * (acc[i][j] - mean[j]);
            }
        }
        for(int j = 0; j < 3; j++)
            var[j] = Math.sqrt(var[j]/k);

        System.out.format("[Stat]Accuracy:\n-- all: %.3f+/-%.3f\n-- phrase: %.3f+/-%.3f\n-- out: %.3f+/-%.3f\n",
                mean[0], var[0], mean[1], var[1], mean[2], var[2]);

    }

}
