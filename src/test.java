import gLearner.SeqAnalyzer;
import gLearner.Sequence;
import gnu.trove.TObjectIntHashMap;

import javax.swing.plaf.synth.SynthDesktopIconUI;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.*;

public class test {
    public static void main(String args[]){
        ModelParameter para = new ModelParameter(args);
        System.setProperty("java.util.logging.config.file", "./log.property");
        /***** experiment setting *****/

        /***** data setting *****/
        String strPath = String.format("%s/%s/%s_string.txt", para.m_prefix, para.m_source, para.m_source);
        String lblPath = String.format("%s/%s/%s_label.txt", para.m_prefix, para.m_source, para.m_source);
        /**** load string and label data ****/
        SeqAnalyzer seqAnalyzer = new SeqAnalyzer(para.m_source);
        seqAnalyzer.loadSequence(strPath, lblPath, 4000, "new");
        ArrayList<Sequence> seqs = seqAnalyzer.getSequences();
        Collections.shuffle(seqs);
        String newstrPath = String.format("%s/%s/%s_string_shuffle.txt", para.m_prefix, para.m_source, para.m_source);
        String newlblPath = String.format("%s/%s/%s_label_shuffle.txt", para.m_prefix, para.m_source, para.m_source);
        try{
            //output result
            File strfile = new File(newstrPath);
            File lblfile = new File(newlblPath);
            BufferedWriter writer1 = new BufferedWriter(new FileWriter(strfile));
            BufferedWriter writer2 = new BufferedWriter(new FileWriter(lblfile));
            for(int i = 0; i < seqs.size(); i++){
                String[] strs = seqs.get(i).getTokens();
                String[] lbls = seqs.get(i).getLabels();
                for(int j = 0; j < strs.length; j++){
                    writer1.write(strs[j]);
                    writer2.write(lbls[j] + ",");
                }
                writer1.write("\n");
                writer2.write("\n");
            }
            writer1.close();
            writer2.close();
        } catch (IOException e){
            e.printStackTrace();
        }

    }
}
