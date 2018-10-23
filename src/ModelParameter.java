import java.util.ArrayList;

/**
 * Created by lulin on 10/22/18.
 */
public class ModelParameter {
    public int m_crossV = 5;
    public int m_samplesize = 100;
    public String m_prefix = "./data";
    public String m_source = "sod";
    public ArrayList<Integer> m_mask = null;

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
            else if(argv[i-1].equals("-mask")){
                m_mask = new ArrayList<>();
                String[] types = argv[i].split(",");
                for(String type : types)
                    m_mask.add(Integer.valueOf(type));
            }
            else
                System.exit(1);
        }
    }
}

