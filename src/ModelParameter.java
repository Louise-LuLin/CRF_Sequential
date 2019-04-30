import java.util.ArrayList;

/**
 * Created by lulin on 10/22/18.
 */
public class ModelParameter {
    public int m_crossV = 5;
    public int m_samplesize = 100;
    public String m_prefix = "./data";
    public String m_source = "sod";
    public ArrayList<Integer> m_mask = new ArrayList<>();
    public int m_iterMax = 30;
    public int m_train_k = 30;
    public int m_test_k = 200;
    public int m_query_k = 100;
    public int m_tuple_k = 0;
    public int m_budget_k = 100;
    //LC: least confidence
    //M: margin
    //SE: subsequence entropy
    public String m_model = "LC";

    public ModelParameter(String argv[]){

        int i;

        //parse options
        for(i=0;i<argv.length;i++) {
            if(argv[i].charAt(0) != '-')
                break;
            else if(++i>=argv.length)
                System.exit(1);
            else if (argv[i-1].equals("-prefix"))
                m_prefix = argv[i];
            else if (argv[i-1].equals("-source"))
                m_source = argv[i];
            else if(argv[i-1].equals("-crossV"))
                m_crossV = Integer.valueOf(argv[i]);
            else if(argv[i-1].equals("-samplesize"))
                m_samplesize = Integer.valueOf(argv[i]);
            else if(argv[i-1].equals("-iter"))
                m_iterMax = Integer.valueOf(argv[i]);
            else if(argv[i-1].equals("-traink"))
                m_train_k = Integer.valueOf(argv[i]);
            else if(argv[i-1].equals("-testk"))
                m_test_k = Integer.valueOf(argv[i]);
            else if(argv[i-1].equals("-queryk"))
                m_query_k = Integer.valueOf(argv[i]);
            else if(argv[i-1].equals("-tuplek"))
                m_tuple_k = Integer.valueOf(argv[i]);
            else if(argv[i-1].equals("-budgetk"))
                m_budget_k = Integer.valueOf(argv[i]);
            else if (argv[i-1].equals("-model"))
                m_model = argv[i];
            else if(argv[i-1].equals("-mask")){
                String[] types = argv[i].split(",");
                for(String type : types)
                    m_mask.add(Integer.valueOf(type));
            }
            else
                System.exit(1);
        }
    }
}

