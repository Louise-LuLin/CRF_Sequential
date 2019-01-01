package gLearner;

import cc.mallet.grmm.inference.Inferencer;
import cc.mallet.grmm.inference.LoopyBP;
import cc.mallet.grmm.types.*;
//import cc.mallet.optimize.LimitedMemoryBFGS;
import edu.umass.cs.mallet.base.maximize.LimitedMemoryBFGS;
import edu.umass.cs.mallet.base.maximize.Maximizable;
import edu.umass.cs.mallet.base.maximize.Maximizer;

//import edu.umass.cs.mallet.base.maximize.LimitedMemoryBFGS;
//import edu.umass.cs.mallet.base.maximize.Maximizable;
//import edu.umass.cs.mallet.base.maximize.Maximizer;
//import edu.umass.cs.mallet.grmm.inference.Inferencer;
//import edu.umass.cs.mallet.grmm.inference.JunctionTreeInferencer;
//import edu.umass.cs.mallet.grmm.inference.LoopyBP;
//import edu.umass.cs.mallet.grmm.types.*;
//import edu.umass.cs.mallet.grmm.types.Assignment;
//import edu.umass.cs.mallet.grmm.types.FactorGraph;

import java.io.*;
import java.util.*;

public class GraphLearner implements Maximizable.ByGradient{

    private double[] m_weights; //weights for each feature, to be optimized
    private double[] m_constraints; //observed counts for each feature (over cliques) in the training sample set (x, y)
    private double[] m_exptectations; //expected counts for each feature (over cliques) based on the training sample set (X)

    Maximizer.ByGradient m_maxer = new LimitedMemoryBFGS();//gradient based optimizer

    ArrayList<String4Learning> m_trainSampleSet = null; // training sample (table factors, feaType, labels)
    ArrayList<FactorGraph> m_trainGraphSet = null;
    ArrayList<Assignment> m_trainAssignment = null;
    private int[] m_foldAssign;
    private int m_foldID;

    TreeMap<Integer, Integer> m_featureMap;

    Inferencer m_infer; //inferencer for marginal computation

    boolean m_scaling;
    boolean m_updated;
    boolean m_trained;
    double m_lambda;
    double m_oldLikelihood;
    Random m_rand = new Random();
    boolean m_flag_gd;

    BufferedWriter m_writer;

    GraphLearner(ArrayList<String4Learning> traininglist){
        m_infer = new LoopyBP(50);
//        m_infer = new JunctionTreeInferencer();

        int featureDim = setTrainingSet(traininglist);
        m_weights = new double[featureDim];
        m_constraints = new double[featureDim];
        m_exptectations = new double[featureDim];

        //training parameters
        m_scaling = false;
        m_updated = true;
        m_trained = false;
        m_rand = new Random();

        m_lambda = 2.0;//L2 regularization
        m_oldLikelihood = -Double.MAX_EXPONENT;//init value

        m_flag_gd = false;
    }

    private int setTrainingSet(ArrayList<String4Learning> traininglist){
        m_trainSampleSet = traininglist;
        m_featureMap = new TreeMap<>();
        for(String4Learning sample : traininglist){
            for(Integer feature : sample.featureType){
                if (!m_featureMap.containsKey(feature)){
                    m_featureMap.put(feature, m_featureMap.size());
                }
            }
        }
        System.out.println("[Stat]Feature size: " + m_featureMap.size());
        //System.out.println(Math.sqrt(m_featureMap.size()-2));
        return m_featureMap.size();
    }

    public Map<Integer, Double> getWeights(){
        Map<Integer, Double> weights = new TreeMap<>();
        Iterator<Map.Entry<Integer, Integer>> it = m_featureMap.entrySet().iterator();
        Map.Entry<Integer, Integer> pairs;
        while (it.hasNext()) {
            pairs = it.next();
            weights.put(pairs.getKey(), m_weights[pairs.getValue()]);
        }
        return weights;
    }

    @Override
    public int getNumParameters() {
        return m_weights.length;
    }

    @Override
    public double getParameter(int index){
        if (index<m_weights.length)
            return m_weights[index];
        else
            return 0;
    }

    @Override
    public void getParameters(double[] buffer){
        if ( buffer.length != m_weights.length )
            buffer = new double[m_weights.length];
        System.arraycopy(m_weights, 0, buffer, 0, m_weights.length);
    }

    @Override
    public void setParameter(int index, double value){
        if ( index<m_weights.length )
            m_weights[index] = value;
        else{
            double[] weights = new double[index+1];
            System.arraycopy(weights, 0, m_weights, 0, m_weights.length);
            weights[index] = value;
            m_weights = weights;
        }
    }

    @Override
    public void setParameters(double[] params){
        if( params.length != m_weights.length ){
            m_weights = new double[params.length];
        }
//        Map<Integer, Double> weights = getWeights();
//        try {
//            for(Integer fea : weights.keySet())
//                m_writer.write(fea.toString() + " : " + weights.get(fea).toString() + " ");
//            m_writer.write("\t" + m_oldLikelihood + "\n");
//            m_writer.flush();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
        System.arraycopy(params, 0, m_weights, 0, m_weights.length);
        m_updated = true;
    }

    @Override
    public double getValue() {
        if ( m_updated ){
            buildFactorGraphs();
            m_updated = false;
        }
        else
            return m_oldLikelihood;

        double tmp;
        FactorGraph graph;
        Assignment assign;

        m_oldLikelihood = 0;
        for(int feaID=0; feaID<m_weights.length; feaID++) {
            m_oldLikelihood -= m_weights[feaID] * m_weights[feaID];
        }
        m_oldLikelihood *= m_lambda/2;    //L2 penalty

        double scale = m_scaling ? (1.0/m_trainSampleSet.size()) : 1.0;
        for(int stringID=0; stringID<m_trainGraphSet.size(); stringID++) {
            assign = m_trainAssignment.get(stringID);
            graph = m_trainGraphSet.get(stringID);
            m_infer.computeMarginals(graph);
            tmp = m_infer.lookupLogJoint(assign);
            if( Double.isNaN(tmp) || tmp>0 ) {
                System.err.println("likelihood failed with " + tmp + "!");
                //System.out.println("Assignment: " + assign);
            }else {
                m_oldLikelihood += tmp * scale;
            }
        }

        System.out.println("[Info]Log-likelihood " + m_oldLikelihood);
        return m_oldLikelihood;//negative log-likelihood or log-likelihood?
    }

    @Override
    public void getValueGradient(double[] buffer) {
        FactorGraph graph;
        Factor factor, ptl;
        String4Learning sample;
        int feaID;

        double feaValue, prob;

        for(feaID=0; feaID<m_exptectations.length; feaID++)
            m_exptectations[feaID] = 0; //clear the SS

        for(int sampleID=0; sampleID<m_trainSampleSet.size(); sampleID++) {
            graph = m_trainGraphSet.get(sampleID);
            sample = m_trainSampleSet.get(sampleID);

            m_infer.computeMarginals(graph);//begin to collect the expectations
            for(int index=0; index<sample.factorList.size(); index++) {
                factor = sample.factorList.get(index);
                ptl = m_infer.lookupMarginal(factor.varSet());
                feaID = m_featureMap.get(sample.featureType.get(index)).intValue();

                for(AssignmentIterator it = ptl.assignmentIterator (); it.hasNext(); it.advance()){
                    Assignment assn = it.assignment ();
                    prob = ptl.value(assn); //get the marginal probability for this local configuration
                    feaValue = factor.logValue(assn); //feature value;
                    m_exptectations[feaID] += feaValue * prob;
                }
            }
        }

        double scale = m_scaling ? (1.0/m_trainSampleSet.size()) : 1.0;
        for(feaID=0; feaID<m_weights.length; feaID++){
            buffer[feaID] = scale * (m_constraints[feaID] - m_exptectations[feaID]) - (m_weights[feaID] * m_lambda);
//            buffer[feaID] = scale * (m_constraints[feaID] - m_exptectations[feaID]);
        }
//        System.out.format("[Info]Gradient: %f\n", Utils.L2Norm(buffer));
    }

    void initWeight(){
        for(int i=0; i<m_weights.length; i++)
            m_weights[i] = m_rand.nextDouble();
    }

    void initialization(boolean initWeight){
        FactorGraph graph;
        String4Learning sample;
        Assignment assign;
        Factor factor;
        int[] assignment;
        int feaID;
        double feaValue;

        //build the initial factor graph
        if(initWeight){
            initWeight();
        }
        buildFactorGraphs();

        //collect the feature counts in the training set
        m_trainAssignment = new ArrayList<>(m_trainSampleSet.size());
        for(int sampleID=0; sampleID<m_trainSampleSet.size(); sampleID++){
            graph = m_trainGraphSet.get(sampleID);
            sample = m_trainSampleSet.get(sampleID);

            //get the graph's assignment over the graph
            assignment = new int[sample.labelList.length];
            for(int i=0; i<sample.labelList.length; i++)
                assignment[i] = sample.labelList[i];
            //System.out.println(graph.numVariables());
            //for(int i=0;i<assignment.length;i++)System.out.println(assignment[i]);
            assign = new Assignment(graph, assignment);
            m_trainAssignment.add(assign);

            for(int i=0; i<sample.factorList.size(); i++){
                factor = sample.factorList.get(i);
                feaID = m_featureMap.get(sample.featureType.get(i)).intValue();
                feaValue = factor.logValue(assign);
                m_constraints[feaID] += feaValue;   // valid for binary feature only?
            }
        }
        System.out.println("Finish collecting sufficient statistics...");

    }

    void buildFactorGraphs(){

        FactorGraph stringGraph;
        String4Learning tmpString;
        Factor factor;
        VarSet clique;
        int index, feaID, stringID;
        HashMap<VarSet, Integer> factorIndex = new HashMap<>();
        Vector<Factor> factorList = new Vector<>();

        // Convert and cache the factors in each string into a factor graph.
        boolean init = (m_trainGraphSet == null);
        // Initialize for the first time.
        if(init){
            //m_trainGraphSet = new ArrayList<>(m_trainSampleSet.size());
            m_trainGraphSet = new ArrayList<>();
        }

        for(stringID=0; stringID<m_trainSampleSet.size(); stringID++){

            tmpString = m_trainSampleSet.get(stringID);
            if(init){
                stringGraph = new FactorGraph();
            }else{
                stringGraph = m_trainGraphSet.get(stringID);
                stringGraph.clear();    //is it safe?
            }
            factorIndex.clear();
            factorList.clear();

            for(index=0; index<tmpString.factorList.size(); index++){
                factor = tmpString.factorList.get(index);
                Factor copy = factor.duplicate();
                feaID = m_featureMap.get(tmpString.featureType.get(index)); // feature ID corresponding to its weight

                copy.exponentiate( m_weights[feaID] );  // potential = feature * weight
                clique = copy.varSet(); // to deal with factors defined over the same clique
                if( factorIndex.containsKey(clique) ){
                    feaID = factorIndex.get(clique);
                    factor = factorList.get(feaID);
                    factor.multiplyBy(copy);
                } else {
                    factorIndex.put(clique, factorList.size());
                    factorList.add(copy);
                }
            }

            //construct the graph
            for(index=0; index<factorList.size(); index++)
                stringGraph.addFactor(factorList.get(index));
            if (init) {
                m_trainGraphSet.add(stringGraph);
            }

        }
        if (init) {
            System.out.println("Finish building " + m_trainGraphSet.size() + "factor graphs...");
        }

    }

    public ArrayList<ArrayList<Integer>> doTraining(int maxIter) {
        initialization(true);   //build the initial factor graphs and collect the constraints from data
        double oldLikelihood = getValue(), likelihood;  // initial likelihood

        if(!m_flag_gd) {
            try {
                if (!m_maxer.maximize(this, maxIter)) {  //if failed, try it again
                    System.err.println("Optimizer fails to converge!");
                    ((LimitedMemoryBFGS) m_maxer).reset();
                    m_maxer.maximize(this, maxIter);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            int iter= 0;
            double[] param = new double[m_weights.length];
            double[] buffer = new double[m_weights.length];

            do{
                iter += 1;
                getParameters(param);
                getValueGradient(buffer);
                for(int i = 0; i < m_weights.length; i++)
                    param[i] += 0.1 * buffer[i];
                setParameters(param);

                likelihood = getValue();
//                System.out.format("[Info]Log-likelihood: %f", likelihood);
            } while (iter <= maxIter);
        }

        likelihood = getValue();
        m_trained = true;

        System.out.println("Training process start, with likelihood " + oldLikelihood);
        System.out.println("Training process finish, with likelihood " + likelihood);

        ArrayList<ArrayList<Integer>> testPrediction = new ArrayList<>();
        for(FactorGraph graph:m_trainGraphSet){
            testPrediction.add(doTesting(graph));
        }

        return testPrediction;
    }

    public void LoadWeights(String filename){
        try {
            BufferedReader reader = new BufferedReader(new FileReader(new File(filename)));
            String tmpTxt;
            String[] feature;
            int feaPtx;
            Integer fea;
            while( (tmpTxt=reader.readLine()) != null ){
                feature = tmpTxt.split(" : ");
                fea = new Integer(feature[0]);
                if( m_featureMap.containsKey(fea) )
                {
                    feaPtx = m_featureMap.get(fea).intValue();
                    m_weights[feaPtx] = Double.valueOf(feature[1]);
                }
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void SaveWeights(String filename){
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(new File(filename)));//append mode
            Map<Integer, Double> weights = getWeights();
            for(Integer fea : weights.keySet())
                writer.write(fea.toString() + " : " + weights.get(fea).toString() + "\n");
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public double calcConfidence(FactorGraph graph){
        double confidence = 0;
        AssignmentIterator it;
        Factor ptl;
        Variable variable;
        int varSize, var;
        double max;

//        Inferencer m_infer = TRP.createForMaxProduct();
        Inferencer m_infer = LoopyBP.createForMaxProduct();

        varSize = graph.numVariables();
        m_infer.computeMarginals(graph);  //begin to collect the expectations

        for(var=0; var<varSize; var++) {
            //retrieve the MAP configuration
            variable = graph.get(var);
            ptl = m_infer.lookupMarginal(variable);
            max = -Double.MAX_VALUE;
            for (it = ptl.assignmentIterator(); it.hasNext(); it.next()) {
                //System.out.println(ptl.value(it));
                if (ptl.value(it)>max) {
                    max = ptl.value(it);
                }
            }
            confidence += max;
        }

        return confidence/varSize;
    }

    public double[] calcTupleConfidence(FactorGraph graph, int k){
        AssignmentIterator it;
        Factor ptl;
        Collection<Variable> variables = new ArrayList<>();
        int varSize, var, i;
        double max;
        double min=Double.MAX_VALUE, tmp;

//        Inferencer m_infer = TRP.createForMaxProduct();
        Inferencer m_infer = LoopyBP.createForMaxProduct();

        varSize = graph.numVariables();
        m_infer.computeMarginals(graph);  //begin to collect the expectations

        double[] tuple_confidence = new double[varSize-k+1];

        for(var=0; var<varSize-k+1; var++) {
            //retrieve the MAP configuration
            variables.clear();
            for(i=var; i < k; i++)
                variables.add(graph.get(var));

            //get all the varset containing all the variables
            HashVarSet c = new HashVarSet();
            Collection adjFactors = graph.allFactorsContaining(variables);
            for (Iterator adjf = adjFactors.iterator (); adjf.hasNext ();) {
                Factor factor = (Factor) adjf.next ();
                c.addAll (factor.varSet ());
            }

            ptl = m_infer.lookupMarginal(c);
            max = -Double.MAX_VALUE;
            for (it = ptl.assignmentIterator(); it.hasNext(); it.next()) {
                //System.out.println(ptl.value(it));
                if (ptl.value(it)>max) {
                    max = ptl.value(it);
                }
            }
            tuple_confidence[var] = max/k;
        }

        return tuple_confidence;
    }

    public ArrayList<Integer> doTesting(FactorGraph graph){

        AssignmentIterator it;
        Factor ptl;
        Variable variable;
        int varSize, var, labelID = 0;
        double max;
        ArrayList<Integer> pred = new ArrayList<>();

//        Inferencer m_infer = TRP.createForMaxProduct();
        Inferencer m_infer = LoopyBP.createForMaxProduct();

        varSize = graph.numVariables();
        m_infer.computeMarginals(graph);  //begin to collect the expectations

        for(var=0; var<varSize; var++) {
            //retrieve the MAP configuration
            variable = graph.get(var);
            ptl = m_infer.lookupMarginal(variable);
            max = -Double.MAX_VALUE;
            for (it = ptl.assignmentIterator(); it.hasNext(); it.next()) {
                //System.out.println(ptl.value(it));
                if (ptl.value(it)>max) {
                    max = ptl.value(it);
                    labelID = it.indexOfCurrentAssn();
                }
            }
            pred.add(labelID);
        }

        return pred;
    }

    // Build a set of factor graphs for the test set.
    public FactorGraph buildFactorGraphs_test(String4Learning tmpString){

        FactorGraph stringGraph = new FactorGraph();
        Factor factor;
        VarSet clique;
        int index, feaID;
        HashMap<VarSet, Integer> factorIndex = new HashMap<>();
        Vector<Factor> factorList = new Vector<>();

        factorIndex.clear();
        factorList.clear();

        for(index=0; index<tmpString.factorList.size(); index++){
            factor = tmpString.factorList.get(index);
            Factor copy = factor.duplicate();
//            if(m_featureMap.containsKey(tmpString.featureType.get(index))) {
//                feaID = m_featureMap.get(tmpString.featureType.get(index)); // feature ID corresponding to its weight
//                copy.exponentiate(m_weights[feaID]);  // potential = feature * weight
//            }else{
//                copy.exponentiate(0);
//            }
            copy.exponentiate(1.0);
            clique = copy.varSet(); // to deal with factors defined over the same clique
            if( factorIndex.containsKey(clique) ){
                feaID = factorIndex.get(clique);
                factor = factorList.get(feaID);
                factor.multiplyBy(copy);
            } else {
                factorIndex.put(clique, factorList.size());
                factorList.add(copy);
            }
        }

        //construct the graph
        for(index=0; index<factorList.size(); index++)
            stringGraph.addFactor(factorList.get(index));

        return stringGraph;

    }

    private void allocateFoldAssignment(int fold)
    {
        m_foldAssign = new int[m_trainSampleSet.size()];
        for(int i=0; i<m_trainSampleSet.size(); i++){
            m_foldAssign[i] = m_rand.nextInt(fold);
        }
    }

    public Vector<Double> getMeanDev(Vector<Double> stat)
    {
        Vector<Double> result = new Vector<Double>();
        double mean = 0, dev = 0;
        for(Double value : stat)
            mean += value.doubleValue();
        mean /= stat.size();
        result.add(mean);

        for(Double value : stat)
            dev += (value.doubleValue() - mean) * (value.doubleValue() - mean);
        dev = Math.sqrt(dev/stat.size());
        result.add(dev);
        return result;
    }

}
