import gLearner.*;

import java.util.HashMap;


public class Main {

    public static void main(String args[]){
        ModelParameter para = new ModelParameter(args);
        System.setProperty("java.util.logging.config.file", "./log.property");
        /***** experiment setting *****/

        /***** data setting *****/
//        String strPath = String.format("%s/%s/%s_string.txt", para.m_prefix, para.m_source, para.m_source);
//        String lblPath = String.format("%s/%s/%s_label.txt", para.m_prefix, para.m_source, para.m_source);
        String tkNamePath = String.format("%s/%s/tokenName.txt", para.m_prefix, para.m_source);
        String lbNamePath = String.format("%s/%s/labelName.txt", para.m_prefix, para.m_source);

        String strPath = String.format("%s/%s/train_string.txt", para.m_prefix, para.m_source);
        String lblPath = String.format("%s/%s/train_label.txt", para.m_prefix, para.m_source);
        String strPath_test = String.format("%s/%s/test_string.txt", para.m_prefix, para.m_source);
        String lblPath_test = String.format("%s/%s/test_label.txt", para.m_prefix, para.m_source);


        /**** load string and label data ****/
        SeqAnalyzer seqAnalyzer = new SeqAnalyzer(para.m_source);
        seqAnalyzer.loadSequence(strPath, lblPath, para.m_samplesize, "new");
        seqAnalyzer.loadSequence(strPath_test, lblPath_test, para.m_samplesize, "concatenate");
        seqAnalyzer.saveTokenNames(tkNamePath);
        seqAnalyzer.saveLabelNames(lbNamePath);

        /**** construct features ****/
        /* 1,3,5,7,9 are isDigit types
         * 0,2,4,6,8 are token itself
         * 10 is edge y_{t}y_{t+1}
         * 11,12,13,14 are position features denoting which part x_t is at
         * 15,16,17,18 are position features denoting which part x_t-1 is at
         * 19,20,21,22 are position features denoting which part x_t+1 is at
         */
        HashMap<Integer, Boolean> featureMasks = new HashMap<Integer, Boolean>();
        for(Integer type : para.m_mask){
            featureMasks.put(type, Boolean.TRUE);
        }
        seqAnalyzer.setMask(featureMasks);

        CRF crfModel = new CRF(seqAnalyzer);
//        crfModel.activeLearning(String.format("%s/%s", para.m_prefix, para.m_source),
//                para.m_iterMax, para.m_train_k, para.m_test_k, para.m_query_k, para.m_tuple_k, para.m_budget_k, para.m_model);
        crfModel.crossValidation(para.m_crossV, String.format("%s/%s", para.m_prefix, para.m_source), para.m_iterMax);
        crfModel.oneFold(81, para.m_iterMax);
    }

}
