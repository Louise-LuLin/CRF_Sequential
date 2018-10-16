public class Main {

    public static void main(String args[]){
        /***** experiment setting *****/
        int crossV = 2;

        /***** data setting *****/
        String prefix = "./data";
        String source = "sod";
        String strPath = String.format("%s/%s/%s_string.txt", prefix, source, source);
        String tkNamePath = String.format("%s/%s/tokenName.txt", prefix, source);
        String lblPath = String.format("%s/%s/%s_label.txt", prefix, source, source);
        String lbNamePath = String.format("%s/%s/labelName.txt", prefix, source);

        /**** load string and label data ****/
        SeqAnalyzer seqAnalyzer = new SeqAnalyzer(source);
        seqAnalyzer.loadString(strPath, 60);
        seqAnalyzer.loadLabel(lblPath, 60);
        seqAnalyzer.saveTokenNames(tkNamePath);
        seqAnalyzer.saveLabelNames(lbNamePath);
        /**** construct features ****/

        CRF crfModel = new CRF(seqAnalyzer);
        crfModel.crossValidation(crossV, String.format("%s/%s", prefix, source));
    }

}
