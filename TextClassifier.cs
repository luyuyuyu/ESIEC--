using System;
using System.IO;
using System.Text;
using System.Linq;
using JiebaNet.Segmenter;
using Accord.MachineLearning;
using Accord.MachineLearning.Bayes;
using Accord.Statistics.Distributions.Univariate;
using Accord.Statistics.Distributions.Fitting;

class Document {
    public static JiebaSegmenter Segmenter = new JiebaSegmenter();

    public int ID;

    // ====input====

    // can we find it on DianPing.com?
    public bool openNow;
    public bool openedOnce;
    public bool noInformation;

    //did the reporter finish GPS positioning? 
    public bool finish;

    // note
    public string note;

    // Note separated by word
    public string[] noteSparated;

    // eigenvectors from notes
    public double[] noteEigenvectors;

    // all eigenvectors
    public double[] eigenvectors;

    // ====output====

    public int contact;
    public static string[] explaination of contact = {
        "打不通电话", "打通了但拒访", "打通了但有事", "其他"//"cannot phone them", "can contact by phone but rejected"
    };                                                //"can contact but the are busy and cannot be researched now" "else"

    public int 存在状况;
    public static string[] 存在状况说明 = {
        "存在至今", "曾经存在", "从未存在", "不知道"
    };

    public bool 数据有效;

    // ====程序代码====

    public Document(int id) {
        ID = id;
        using (var file = new StreamReader($"Data/Input{id}.txt")) {
            switch (file.ReadLine()) {
                case "大众点评上能否查询到该企业？能查到正在营业":
                    能查到正在营业 = true; break;
                case "大众点评上能否查询到该企业？能查到曾经营业":
                    能查到曾经营业 = true; break;
                case "大众点评上能否查询到该企业？无信息":
                    无营业信息 = true; break;
                default:
                    throw new FormatException($"请检查 Input{id}.txt 的格式");
            }
            switch (file.ReadLine()) {
                case "队员是否有成功的GPS定位？有":
                    有GPS定位 = true; break;
                case "队员是否有成功的GPS定位？没有":
                    有GPS定位 = false; break;
                default:
                    throw new FormatException($"请检查 Input{id}.txt 的格式");
            }
            if (file.ReadLine().Length != 0) {
                throw new FormatException($"请检查 Input{id}.txt 的格式");
            }
            var comment = new StringBuilder();
            while (true) {
                var line = file.ReadLine();
                if (line == null) break;
                comment.Append(line);
            }
            备注文本 = comment.ToString();
            备注词汇 = Segmenter.Cut(备注文本)
                .Where(k => !char.IsPunctuation(k[0])).ToArray();
        }
    }

    public void ParseOutput() {
        using (var file = new StreamReader($"Data/Output{ID}.txt")) {
            var line1 = file.ReadLine();
            if (!line1.StartsWith("通讯情况？")) {
                throw new FormatException($"请检查 Output{ID}.txt 的格式");
            }
            通讯情况 = Array.FindIndex(通讯情况说明, k => line1.EndsWith(k));
            if (通讯情况 == -1) {
                throw new FormatException($"请检查 Output{ID}.txt 的格式");
            }
            var line2 = file.ReadLine();
            if (!line2.StartsWith("存在状况？")) {
                throw new FormatException($"请检查 Output{ID}.txt 的格式");
            }
            存在状况 = Array.FindIndex(存在状况说明, k => line2.EndsWith(k));
            if (存在状况 == -1) {
                throw new FormatException($"请检查 Output{ID}.txt 的格式");
            }
            switch (file.ReadLine()) {
                case "数据有效性？有效":
                    数据有效 = true; break;
                case "数据有效性？无效":
                    数据有效 = false; break;
                default:
                    throw new FormatException($"请检查 Output{ID}.txt 的格式");
            }
        }
    }
}

class TextClassifier {
    const int N = 14;

    public static void Main() {
        var documents = new Document[N];
        var words = new string[N][];
        for (int i = 0; i < N; ++i) {
            documents[i] = new Document(i);
            words[i] = documents[i].备注词汇;
        }
        var tfIdf = new TFIDF();
        tfIdf.Learn(words);
        var inputs = new double[N][];
        for (int i = 0; i < N; ++i) {
            documents[i].备注特征向量 = tfIdf.Transform(documents[i].备注词汇);
            documents[i].特征向量 = new double[documents[i].备注特征向量.Length + 4];
            documents[i].特征向量[0] = documents[i].能查到正在营业 ? 1.0 : 0.0;
            documents[i].特征向量[1] = documents[i].能查到曾经营业 ? 1.0 : 0.0;
            documents[i].特征向量[2] = documents[i].无营业信息 ? 1.0 : 0.0;
            documents[i].特征向量[3] = documents[i].有GPS定位 ? 1.0 : 0.0;
            documents[i].备注特征向量.CopyTo(documents[i].特征向量, 4);
            inputs[i] = documents[i].特征向量;
        }
        var outputs通讯情况 = new int[N];
        var outputs存在状况 = new int[N];
        var outputs数据有效 = new int[N];
        for (int i = 0; i < N; ++i) {
            documents[i].ParseOutput();
            outputs通讯情况[i] = documents[i].通讯情况;
            outputs存在状况[i] = documents[i].存在状况;
            outputs数据有效[i] = documents[i].数据有效 ? 1 : 0;
        }
        var teacher1 = new NaiveBayesLearning<NormalDistribution>();
        teacher1.Options.InnerOption = new NormalOptions {
            Regularization = 1e-12
        };
        var teacher2 = new NaiveBayesLearning<NormalDistribution>();
        teacher2.Options.InnerOption = new NormalOptions {
            Regularization = 1e-12
        };
        var teacher3 = new NaiveBayesLearning<NormalDistribution>();
        teacher3.Options.InnerOption = new NormalOptions {
            Regularization = 1e-12
        };
        var model通讯情况 = teacher1.Learn(inputs, outputs通讯情况);
        var model存在状况 = teacher2.Learn(inputs, outputs存在状况);
        var model数据有效 = teacher3.Learn(inputs, outputs数据有效);
        var correct通讯情况 = 0;
        var correct存在状况 = 0;
        var correct数据有效 = 0;
        for (int i = 0; i < N; ++i) {
            var 通讯情况 = model通讯情况.Decide(documents[i].特征向量);
            if (documents[i].通讯情况 == 通讯情况) ++correct通讯情况;
            else Console.WriteLine("Input{0}.txt的通讯情况 你认为:{1} 电脑认为:{2}",
                i, Document.通讯情况说明[documents[i].通讯情况],
                Document.通讯情况说明[通讯情况]);
            var 存在状况 = model存在状况.Decide(documents[i].特征向量);
            if (documents[i].存在状况 == 存在状况) ++correct存在状况;
            else Console.WriteLine("Input{0}.txt的存在状况 你认为:{1} 电脑认为:{2}",
                i, Document.存在状况说明[documents[i].存在状况],
                Document.存在状况说明[存在状况]);
            var 数据有效 = model数据有效.Decide(documents[i].特征向量) == 1;
            if (documents[i].数据有效 == 数据有效) ++correct数据有效;
            else Console.WriteLine("Input{0}.txt的数据有效 你认为:{1} 电脑认为:{2}",
                i, documents[i].数据有效, 数据有效);
        }
        Console.WriteLine("通讯情况准确率: {0:F2} %",
            (double)correct通讯情况 / N * 100);
        Console.WriteLine("存在状况准确率: {0:F2} %",
            (double)correct存在状况 / N * 100);
        Console.WriteLine("数据有效准确率: {0:F2} %",
            (double)correct数据有效 / N * 100);
    }
}
