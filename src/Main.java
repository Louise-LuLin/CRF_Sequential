import gLearner.*;

import java.util.HashMap;


public class Main {

    public static void main(String args[]){
        ModelParameter para = new ModelParameter(args);
        /***** experiment setting *****/
        int crossV = 2;

        /***** data setting *****/
        String strPath = String.format("%s/%s/%s_string.txt", para.m_prefix, para.m_source, para.m_source);
        String tkNamePath = String.format("%s/%s/tokenName.txt", para.m_prefix, para.m_source);
        String lblPath = String.format("%s/%s/%s_label.txt", para.m_prefix, para.m_source, para.m_source);
        String lbNamePath = String.format("%s/%s/labelName.txt", para.m_prefix, para.m_source);

        /**** load string and label data ****/
        SeqAnalyzer seqAnalyzer = new SeqAnalyzer(para.m_source);
        seqAnalyzer.loadString(strPath, para.m_samplesize);
        seqAnalyzer.loadLabel(lblPath, para.m_samplesize);
        seqAnalyzer.saveTokenNames(tkNamePath);
        seqAnalyzer.saveLabelNames(lbNamePath);

        /**** construct features ****/
        /* 1,3,5,7,9 are isDigit types
         * 0,2,4,6,8 are token itself
         * 10 is edge y_{t}y_{t+1}
         */
        HashMap<Integer, Boolean> featureMasks = new HashMap<Integer, Boolean>();
        for(Integer type : para.m_mask){
            featureMasks.put(type, new Boolean(false));
        }
        seqAnalyzer.setMask(featureMasks);

        CRF crfModel = new CRF(seqAnalyzer);
        crfModel.crossValidation(para.m_crossV, String.format("%s/%s", para.m_prefix, para.m_source));
    }

}
