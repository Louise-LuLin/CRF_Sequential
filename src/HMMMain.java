
import gLearner.*;

import java.util.*;

public class HMMMain {

    public static void main(String args[]){
        ModelParameter para = new ModelParameter(args);
        System.setProperty("java.util.logging.config.file", "./log.property");
        /***** experiment setting *****/

        /***** data setting *****/
        String strPath = String.format("%s/%s/%s_string_shuffle.txt", para.m_prefix, para.m_source, para.m_source);
        String lblPath = String.format("%s/%s/%s_label_shuffle.txt", para.m_prefix, para.m_source, para.m_source);
        String tkNamePath = String.format("%s/%s/tokenName.txt", para.m_prefix, para.m_source);
        String lbNamePath = String.format("%s/%s/labelName.txt", para.m_prefix, para.m_source);

//        String strPath = String.format("%s/%s/train_string.txt", para.m_prefix, para.m_source);
//        String lblPath = String.format("%s/%s/train_label.txt", para.m_prefix, para.m_source);
//        String strPath_test = String.format("%s/%s/test_string.txt", para.m_prefix, para.m_source);
//        String lblPath_test = String.format("%s/%s/test_label.txt", para.m_prefix, para.m_source);

        /**** load string and label data ****/
        SeqAnalyzer seqAnalyzer = new SeqAnalyzer(para.m_source);
        seqAnalyzer.loadSequence(strPath, lblPath, para.m_samplesize, "new");
//        seqAnalyzer.loadSequence(strPath_test, lblPath_test, para.m_samplesize, "concatenate");
        seqAnalyzer.saveTokenNames(tkNamePath);
        seqAnalyzer.saveLabelNames(lbNamePath);

        // train HMM using MLE
        HMM hmmTagger = new HMM();

        ArrayList<Sequence> seqs = seqAnalyzer.getSequences();

        hmmTagger.activeLearning(seqs, para.m_train_k, para.m_query_k, para.m_test_k, String.format("%s/%s", para.m_prefix, para.m_source),
        para.m_tuple_k, 500, "margin2");


    }

}
