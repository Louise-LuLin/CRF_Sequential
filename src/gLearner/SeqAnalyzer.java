package gLearner;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import edu.umass.cs.mallet.base.types.SparseMatrixn;
import edu.umass.cs.mallet.grmm.types.Factor;
import edu.umass.cs.mallet.grmm.types.LogTableFactor;
import edu.umass.cs.mallet.grmm.types.Variable;

public class SeqAnalyzer {
    private String m_source;
    
    private ArrayList<String> m_labelNames;//text content of the labels Y
    private HashMap<String, Integer> m_labelNameIndex;//index of the labels Y

    private ArrayList<String> m_tokenNames;//text content of the words X
    private HashMap<String, Integer> m_tokenNameIndex;//index of the words X

    //why do not we create a structure for each input string (e.g., sentence or point name)
    private ArrayList<Sequence> m_seqList;

    private Map<Integer, Boolean> m_mask;

    public SeqAnalyzer(String source){
        this.m_source = source;
        
        m_labelNames = new ArrayList<String>();
        m_labelNameIndex = new HashMap<String, Integer>();
        
        m_tokenNames = new ArrayList<String>();
        m_tokenNameIndex = new HashMap<String, Integer>();
        
        m_seqList = new ArrayList<Sequence>();
        m_mask = null;
    }

    public String getSource() { return m_source; } //where we load the data

    public void setMask(Map<Integer, Boolean> masks) { m_mask = masks; }

    public ArrayList<String> getLabelNames(){ return this.m_labelNames; }

    public HashMap<String, Integer> getLabelNameIndex(){ return this.m_labelNameIndex; }

    public ArrayList<String> getStrings(){ 
    	ArrayList<String> strList = new ArrayList<String>();
    	
    	for(Sequence seq:m_seqList) 
    		strList.add(seq.m_content);
    			
    	return strList; 
    }

    public ArrayList<int[]> getLabels() { 
    	ArrayList<int[]> labelList = new ArrayList<int[]>();
    	
    	for(Sequence seq:m_seqList)
    		labelList.add(seq.getLabels());
    	
    	return labelList; 
    }

    public ArrayList<Sequence> getSequences() { return this.m_seqList; }

    public ArrayList<int[]> getTokens() {
    	ArrayList<int[]> tokenList = new ArrayList<int[]>();
    	
    	for(Sequence seq:m_seqList)
    		tokenList.add(seq.getTokens());
    	
    	return tokenList;
    }

    public void saveTokenNames(String filePath){
        try{
            BufferedWriter writer = new BufferedWriter(new FileWriter(new File(filePath)));
            for(int i = 0; i < m_tokenNames.size(); i++)
                writer.write(i + "\t" + m_tokenNames.get(i) + "\n");
            writer.close();
        } catch (IOException e){
            e.printStackTrace();
        }
    }

    public void saveLabelNames(String filePath){
        try{
            BufferedWriter writer = new BufferedWriter(new FileWriter(new File(filePath)));
            for(int i = 0; i < m_labelNames.size(); i++)
                writer.write(i + "\t" + m_labelNames.get(i) + "\n");
            writer.close();
        } catch (IOException e){
            e.printStackTrace();
        }
    }

    public void loadSequence(String filePath, int maxNum){
    	m_seqList.clear();
    	
        // Read training strings.
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line, token;
            
            while ((line = br.readLine()) != null && line.length() > 0) {
            	Sequence seq = new Sequence(line);
                int idx;
                for(int i = 0 ; i < line.length(); i++){
                	token = Character.toString(line.charAt(i));
                    idx = getTokenIndex(token); //dynamically expand tokenNames: each char to string
                    seq.addToken(token, idx);
                }
                
                m_seqList.add(seq);
                if(m_seqList.size() > maxNum)
                    break;
            }
            System.out.format("[Info]token size: %d\n", m_tokenNames.size());

        } catch (Exception e){
            System.err.format("[Err]File %s doesn't exist.\n", filePath);
        }
    }

    //this is a poor design of loading input files, where we have to assume the alignment by line number
    public void loadLabel(String filePath, int maxNum){
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            String[] labels;
            int lineNo = 0;
            
            while ((line = br.readLine()) != null && line.length() > 0) {
            	Sequence seq = m_seqList.get(lineNo);
            	
                labels = line.split(",");
                int tokenIdx = 0;
                for(String s: labels){
                    if(s.isEmpty())
                        continue;
                    seq.assignLabel(tokenIdx++, getLabelIndex(s));//dynamically expand labelNames
                }
                
                if(++lineNo > maxNum)
                    break;
            }
            System.out.format("[Info]Label size: %d\n", m_labelNames.size());

        } catch (Exception e){
            System.err.format("[Err]File %s doesn't exist.\n", filePath);
        }
    }

    //To create the training sequence
    public String4Learning getStr4Learning(Sequence seq, String mode){

        // Each string is stored as an object specifying features as table factors.
        String4Learning str;

        // Feature vectors. Key: feature type indices. The vectors are of
        // the same length equivalent to the string length.
        HashMap<Integer, ArrayList<Double>> node_features;

        // For each training sample, construct a factor graph, and a list of table factors to specify edge
        // and node features.
        //step 1: construct the graph
        int varNodeSize = seq.getLength();
        int[] label_vec = seq.getLabels();
        Variable[] allVars = new Variable[varNodeSize];
        for(int i = 0; i < allVars.length; i++)
            allVars[i] = new Variable(m_labelNames.size()); //each label variable has this many outcomes. But we can reduce this ahead of time based on the nature of our problem?

        ArrayList<Factor> factorList = new ArrayList<Factor>();   // list of table factors for the current string
        ArrayList<Integer> featureType = new ArrayList<Integer>(); // corresponding feature ID for each list of factors

        //step 2: add node features
        node_features = constructNodeFeature(seq.getTokens());
        ArrayList<Double> cur_feature;
        Factor ptl;
        double[] feature_value_arr = new double[m_labelNames.size()];
        double[] trans_feature_arr;
        for(int j = 0; j < varNodeSize; j++) {// for each node/variable

            // note features:
            // token itself:   0 ~ label_size * (token_size + 2) * (num of type%2==0)
            // token is digit: label_size * (token_size + 2) * 5 (=A) ~ A + label_size * (num of type%2==1)
            for (Integer type : node_features.keySet()) {// for each feature type
                //skip feature type with mask
                if(m_mask != null && m_mask.containsKey(type)
                        && !m_mask.get(type))
                    continue;

                cur_feature = node_features.get(type);

                if (type % 2 == 0) { // x_t/t-1...: type=0,2,4,6,8
                    for (int k = 0; k < m_tokenNames.size(); k++) {
                        for(int label_i = 0; label_i < m_labelNames.size(); label_i++) {
                            if (cur_feature.get(j).intValue() == k) {
                                Arrays.fill(feature_value_arr, 1.0);
                                feature_value_arr[label_i] = Math.exp(1.0);
                            }
                            else
                                Arrays.fill(feature_value_arr, 1.0);
                            ptl = LogTableFactor.makeFromValues(new Variable[]{allVars[j]}, feature_value_arr);
                            factorList.add(ptl);
                            featureType.add((m_tokenNames.size() * m_labelNames.size()) * (type / 2) +
                                    m_labelNames.size() * k + label_i);
                        }
                    }
                } else { // is digit: type=1,3,5,7,9
                    for(int label_i = 0; label_i < m_labelNames.size(); label_i++) {
                        Arrays.fill(feature_value_arr, 1.0);
                        feature_value_arr[label_i] = Math.exp(cur_feature.get(j));
                        ptl = LogTableFactor.makeFromValues(new Variable[]{allVars[j]}, feature_value_arr);
                        factorList.add(ptl);
                        featureType.add((m_tokenNames.size() * m_labelNames.size()) * 5 +
                                m_labelNames.size() * (type / 2) + label_i);
                    }
                }
            }
        }

        //step 3: add edge features
        if(m_mask == null || !m_mask.containsKey(10) || !m_mask.get(10)) {
            int node_feature_size = (m_tokenNames.size() * m_labelNames.size()) * 5 // This is incorrect! As we might mask out some node features!!!!
                    + m_labelNames.size() * 5;
            //the size is used to index edge feature such that the index will not overlap for factor graph, we take the largest space for node feature index

            for (int j = 0; j < varNodeSize; j++) {
//                    for (int i = 0; i < m_labelNames.size(); i++) {
//                        for (int k = 0; k < m_labelNames.size(); k++) {
//                            trans_feature_arr = label_transition(i, k);
//                            ptl = LogTableFactor.makeFromValues(
//                                   new Variable[]{allVars[j], allVars[j + 1]}, trans_feature_arr);
//                            factorList.add(ptl);
//                            featureType.add(node_feature_size + i * m_labelNames.size() + k);
//                        }
//                    }

                if(mode.equals("train")) {//train
//                    int curIdx_1 = label_vec[j];
//                    int curIdx_2 = label_vec[j + 1];
//                    trans_feature_arr = label_transition(curIdx_1, curIdx_2);
//
//                    ptl = LogTableFactor.makeFromValues(
//                            new Variable[]{allVars[j], allVars[j + 1]}, trans_feature_arr);
//
//                    factorList.add(ptl);
//                    featureType.add(node_feature_size + 10 + curIdx_1 * m_labelNames.size() + curIdx_2);

                    int curIdx_1;
                    if (j > 0)
                        curIdx_1 = label_vec[j - 1];
                    else
                        curIdx_1 = getLabelIndex("START");

                    trans_feature_arr = new double[m_labelNames.size()];
                    Arrays.fill(trans_feature_arr, 1.0);
                    trans_feature_arr[curIdx_1] = Math.exp(1);
                    ptl = LogTableFactor.makeFromValues(
                            new Variable[]{allVars[j]}, trans_feature_arr);

                    factorList.add(ptl);
                    featureType.add(node_feature_size + 10 + curIdx_1);

                }else{//test
//                    for (int i = 0; i < m_labelNames.size(); i++) {
//                        for (int k = 0; k < m_labelNames.size(); k++) {
//                            trans_feature_arr = label_transition(i, k);
//                            ptl = LogTableFactor.makeFromValues(
//                                    new Variable[]{allVars[j], allVars[j + 1]}, trans_feature_arr);
//                            factorList.add(ptl);
//                            featureType.add(node_feature_size + 10 + i * m_labelNames.size() + k);
//                        }
//                    }

                    for (int i = 0; i < m_labelNames.size(); i++) {
                        int curIdx_1;
                        if (j > 0)
                            curIdx_1 = i;
                        else
                            curIdx_1 = getLabelIndex("START");

                        trans_feature_arr = new double[m_labelNames.size()];
                        Arrays.fill(trans_feature_arr, 1.0);
                        if (j == 0)
                            trans_feature_arr[curIdx_1] = Math.exp(1);

                        ptl = LogTableFactor.makeFromValues(
                                new Variable[]{allVars[j]}, trans_feature_arr);

                        factorList.add(ptl);
                        featureType.add(node_feature_size + 10 + curIdx_1);
                    }
                }
            }
        }

        str = new String4Learning(factorList, featureType, label_vec);

        return str;
    }

    // build the node features
    public HashMap<Integer, ArrayList<Double>> constructNodeFeature(int[] tokens){
        //0: x_t = o
        //1: x_t is digit
        //2: x_t-1 = o
        //3: x_t-1 is digit
        //4: x_t+1 = o
        //5: x_t+1 is digit
        //6: x_t-2 = o
        //7: x_t-2 is digit
        //8: x_t+2 = o
        //9: x_t+2 is digit
        HashMap<Integer, ArrayList<Double>> node_features = new HashMap<>();

        //0: index of x_t
        ArrayList<Double> x_t = new ArrayList<>();
        //1: if x_t is a digit
        ArrayList<Double> x_t_is_digit = new ArrayList<>();

        //2: if x_t-1 = o
        ArrayList<Double> x_t_pre_1 = new ArrayList<>();
        //3: if x_t-1 is a digit
        ArrayList<Double> x_t_pre_1_is_digit = new ArrayList<>();

        //4: if x_t+1 = o
        ArrayList<Double> x_t_next_1 = new ArrayList<>();
        //5: if x_t+1 is a digit
        ArrayList<Double> x_t_next_1_is_digit = new ArrayList<>();

        //6: if x_t-2 = o
        ArrayList<Double> x_t_pre_2 = new ArrayList<>();
        //7: if x_t-2 is a digit
        ArrayList<Double> x_t_pre_2_is_digit = new ArrayList<>();

        //8: if x_t+2 = o
        ArrayList<Double> x_t_next_2 = new ArrayList<>();
        //9: if x_t+2 is a digit
        ArrayList<Double> x_t_next_2_is_digit = new ArrayList<>();

        int curIdx;
        String curToken;
        for(int i = 0;i < tokens.length; i++){
            //x_t
            curIdx = tokens[i];
            curToken = m_tokenNames.get(curIdx);
            x_t.add((double) curIdx);
            //x_t is digit
            if(isDigit(curToken))
                x_t_is_digit.add(1.0);
            else
                x_t_is_digit.add(0.0);

            //x_t-1
            if(i == 0)
                curIdx = getTokenIndex("START");
            else
                curIdx = tokens[i-1];
            curToken = m_tokenNames.get(curIdx);
            x_t_pre_1.add((double) curIdx);
            //x_t-1 is digit
            if(isDigit(curToken))
                x_t_pre_1_is_digit.add(1.0);
            else
                x_t_pre_1_is_digit.add(0.0);

            //x_t-2
            if(i < 2)
                curIdx = getTokenIndex("START");
            else
                curIdx = tokens[i-2];
            curToken = m_tokenNames.get(curIdx);
            x_t_pre_2.add((double) curIdx);
            //x_t-2 is digit
            if(isDigit(curToken))
                x_t_pre_2_is_digit.add(1.0);
            else
                x_t_pre_2_is_digit.add(0.0);

            //x_t+1
            if(i == tokens.length-1)
                curIdx = getTokenIndex("END");
            else
                curIdx = tokens[i+1];
            curToken = m_tokenNames.get(curIdx);
            x_t_next_1.add((double) curIdx);
            //x_t+1 is digit
            if(isDigit(curToken))
                x_t_next_1_is_digit.add(1.0);
            else
                x_t_next_1_is_digit.add(0.0);

            //x_t+2
            if(i == tokens.length-1 || i == tokens.length-2)
                curIdx = getTokenIndex("END");
            else
                curIdx = tokens[i+2];
            curToken = m_tokenNames.get(curIdx);
            x_t_next_2.add((double) curIdx);
            //x_t+2 is digit
            if(isDigit(curToken))
                x_t_next_2_is_digit.add(1.0);
            else
                x_t_next_2_is_digit.add(0.0);

        }

        node_features.put(1,x_t_is_digit);
        node_features.put(3,x_t_pre_1_is_digit);
        node_features.put(5,x_t_next_1_is_digit);
        node_features.put(7,x_t_pre_2_is_digit);
        node_features.put(9,x_t_next_2_is_digit);

        node_features.put(0,x_t);
        node_features.put(2,x_t_pre_1);
        node_features.put(4,x_t_next_1);
        node_features.put(6,x_t_pre_2);
        node_features.put(8,x_t_next_2);

        return node_features;
    }

    // This is the first-order edge feature enumerating all possible label transitions.
    public double[] label_transition(int label1, int label2){
        int len_arr = m_labelNames.size() * m_labelNames.size();
        double[] arr = new double[len_arr];
        Arrays.fill(arr, 1.0);
        arr[label1*m_labelNames.size()+label2] = Math.exp(1.0);
        return arr;
    }

    // Check if a character (token) is a digit.
    private boolean isDigit(String token){
        char token_char = token.charAt(0);
        return Character.isDigit(token_char);
    }

    //get label index and dynamically expand vocabulary if not found
    private int getLabelIndex(String label){
        if(!m_labelNameIndex.containsKey(label)) {
            m_labelNameIndex.put(label, m_labelNames.size());
            m_labelNames.add(label);
        }
        return m_labelNameIndex.get(label);
    }

    //get token index and dynamically expand token list if not found
    private int getTokenIndex(String token){
        if(!m_tokenNameIndex.containsKey(token)) {
            m_tokenNameIndex.put(token, m_tokenNames.size());
            m_tokenNames.add(token);
        }
        return m_tokenNameIndex.get(token);
    }

}
