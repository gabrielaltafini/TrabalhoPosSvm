package com.example.gabriel.gerasvm30;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.ParamGrid;
import org.opencv.ml.SVM;
import org.opencv.xfeatures2d.SURF;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;

import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Vector;
import static org.opencv.ml.Ml.ROW_SAMPLE;

public class MainActivity extends AppCompatActivity {

    Scalar RED = new Scalar(255, 0, 0);
    Scalar GREEN = new Scalar(0, 255, 0);
    private ImageView imageView;
    private Bitmap inputImage; // make bitmap from image resource
    private Vector cinquentaFrente ;
    private final static String TAG = "SVM";
    private  SVM svmpoly;
    private  SVM svmLinear;
    private  SVM svmBRF;

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //inputImage = BitmapFactory.decodeResource(getResources(), R.drawable.nota50);
        setContentView(R.layout.activity_main);
        imageView = (ImageView) this.findViewById(R.id.imageView);
    }

    public void sift() {
        //GeraSVMNovo();
        GeraSVMSurf();
        Log.d(TAG, "AQUI");
        //-----------------------------------------------------
        //saveResults("notas",textoBuffer);
        //-----------------------------------------------------
        //RealizaTeste();
        //RealizaTesteSurf();
        //MostraKey();
    }

    private void RealizaTesteSurf()
    {
        svmBRF = SVM.create();
        File f = new File(getExternalFilesDir(null).getAbsolutePath() + "/svm_trainedb.xml");
        svmBRF  = SVM.load(f.getAbsolutePath());
        if(svmBRF.isTrained())
        {
            Log.i(TAG, "deu bom" );
        }
        else{
            Log.i(TAG, "nao carregou" );
        }

        svmpoly = SVM.create();
        f = new File(getExternalFilesDir(null).getAbsolutePath() + "/svm_trained.xml");
        svmpoly  = SVM.load(f.getAbsolutePath());
        if(svmpoly.isTrained())
        {
            Log.i(TAG, "deu bom" );
        }
        else{
            Log.i(TAG, "nao carregou" );
        }


        svmLinear = SVM.create();
        f = new File(getExternalFilesDir(null).getAbsolutePath() + "/svm_trainedl.xml");
        svmLinear  = SVM.load(f.getAbsolutePath());
        if(svmLinear.isTrained())
        {
            Log.i(TAG, "deu bom" );
        }
        else{
            Log.i(TAG, "nao carregou" );
        }

        Mat rgba = new Mat();
        String vNomeImagem;
        //poly e linear
        //vNomeImagem = "ts20v";
        //poly e linear
        //vNomeImagem = "ts5v";

        //poly e linear
        vNomeImagem = "ts2v";
        int drawable = getResources().getIdentifier(vNomeImagem, "drawable", getPackageName());
        inputImage = BitmapFactory.decodeResource(getResources(), drawable);

        Utils.bitmapToMat(inputImage, rgba);
        MatOfKeyPoint keyPoints = new MatOfKeyPoint();
        Imgproc.cvtColor(rgba, rgba, Imgproc.COLOR_RGB2GRAY);
        //CarregaSVM();
        String  f2 = Processar.identificarFotoSurf(rgba, svmpoly);
        Log.i(TAG, "Poly Valor:" + f2);
        f2 = Processar.identificarFotoSurf(rgba, svmLinear);
        Log.i(TAG, "Liner Valor:" + f2);

        f2 = Processar.identificarFotoSurf(rgba, svmBRF);

        Log.i(TAG, "BRF Valor:" + f2);
        String v = "";
    }

    private void MostraKey()
    {

        Mat rgba = new Mat();
        String vNomeImagem;
        vNomeImagem = "s100_f26";
        int drawable = getResources().getIdentifier(vNomeImagem, "drawable", getPackageName());
        inputImage = BitmapFactory.decodeResource(getResources(), drawable);
        Utils.bitmapToMat(inputImage, rgba);
        Imgproc.cvtColor(rgba, rgba, Imgproc.COLOR_RGB2GRAY);
        MatOfKeyPoint objectKeyPoints = new MatOfKeyPoint();
        SURF detector = SURF.create();
        detector.setHessianThreshold(500);
        detector.setNOctaveLayers(6);
        detector.setNOctaves(3);
        detector.setUpright(false);
        detector.detect(rgba, objectKeyPoints);
        KeyPoint[] keypointsVector = objectKeyPoints.toArray();
        Scalar newKeypointColor = new Scalar(255, 0, 0);
        System.out.println("Drawing key points on object image...");
        Log.d(TAG, "Numero keypoints: " + keypointsVector.length);

        Features2d.drawKeypoints(rgba, objectKeyPoints, rgba, newKeypointColor, 0);
        Utils.matToBitmap(rgba, inputImage);
        imageView.setImageBitmap(inputImage);

        imageView.buildDrawingCache();
        Bitmap bm=((BitmapDrawable)imageView.getDrawable()).getBitmap();
        saveImageFile(bm);
    }


    public String saveImageFile(Bitmap bitmap) {
        FileOutputStream out = null;
        String filename = getFilename();
        try {
            out = new FileOutputStream(filename);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return filename;
    }

    private String getFilename() {
        File file = new File(getExternalFilesDir(null).getAbsolutePath() + "/");
        if (!file.exists()) {
            file.mkdirs();
        }
        String uriSting = (file.getAbsolutePath() + "/"
                + System.currentTimeMillis() + ".jpg");
        return uriSting;
    }

    private void RealizaTeste()
    {
        svmBRF = SVM.create();
        File f = new File(getExternalFilesDir(null).getAbsolutePath() + "/svm_trainedb.xml");
        svmBRF  = SVM.load(f.getAbsolutePath());
        if(svmBRF.isTrained())
        {
            Log.i(TAG, "deu bom" );
        }
        else{
            Log.i(TAG, "nao carregou" );
        }

        svmpoly = SVM.create();
         f = new File(getExternalFilesDir(null).getAbsolutePath() + "/svm_trained.xml");
        svmpoly  = SVM.load(f.getAbsolutePath());
        if(svmpoly.isTrained())
        {
            Log.i(TAG, "deu bom" );
        }
        else{
            Log.i(TAG, "nao carregou" );
        }

        svmLinear = SVM.create();
        f = new File(getExternalFilesDir(null).getAbsolutePath() + "/svm_trainedl.xml");
        svmLinear  = SVM.load(f.getAbsolutePath());
        if(svmLinear.isTrained())
        {
            Log.i(TAG, "deu bom" );
        }
        else{
            Log.i(TAG, "nao carregou" );
        }

        Mat rgba = new Mat();
        String vNomeImagem;
        vNomeImagem = "teste50v";

        int drawable = getResources().getIdentifier(vNomeImagem, "drawable", getPackageName());
        inputImage = BitmapFactory.decodeResource(getResources(), drawable);

        Utils.bitmapToMat(inputImage, rgba);
        MatOfKeyPoint keyPoints = new MatOfKeyPoint();
        Imgproc.cvtColor(rgba, rgba, Imgproc.COLOR_RGB2GRAY);
        //CarregaSVM();
        String  f2 = Processar.identificarFoto(rgba, svmpoly);
        Log.i(TAG, "Valor:" + f2);
        f2 = Processar.identificarFoto(rgba, svmLinear);
        Log.i(TAG, "Valor:" + f2);

        f2 = Processar.identificarFoto(rgba, svmBRF);

        Log.i(TAG, "Valor:" + f2);
        String v = "";
    }

    private void GeraSVMNovo()
    {
        int rowsSize = 360;
        int[] labelArray = new int[rowsSize];
        int index = 0 ;
        int columnSize = (5696);
        Mat trainData = new Mat(rowsSize, columnSize, CvType.CV_32F);
        Mat responses = new Mat(rowsSize, 1, CvType.CV_32S);
        String vComp ="";
        StringBuffer textoBuffer = new StringBuffer("");

        //SURF detector = SURF.create();
        //detector.setHessianThreshold(450);
        //detector.setNOctaveLayers(6);
        //detector.setNOctaves(3);
        //detector.setUpright(false);

        FeatureDetector detector = FeatureDetector.create(FeatureDetector.ORB);
        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);

        Mat rgba = new Mat();
        final String[] VALOR = { "2", "5", "10", "20", "50", "100" };
        final String[] LADO = { "f", "v" };
       // for (int aux = 1 ; aux < 13 ; aux ++ ) {
            for (int n = 11; n < 31; n++) {
                //for (int n = 1; n < 2; n++) {
                for (String v : VALOR) {
                    for (String l : LADO) {
                        String vNomeImagem;
                        vNomeImagem = "s" + v + "_" + l + "" + n;
                        Log.d(TAG, vNomeImagem);
                        int drawable = getResources().getIdentifier(vNomeImagem, "drawable", getPackageName());
                        inputImage = BitmapFactory.decodeResource(getResources(), drawable);
                        Utils.bitmapToMat(inputImage, rgba);
                        MatOfKeyPoint keyPoints = new MatOfKeyPoint();
                        Imgproc.cvtColor(rgba, rgba, Imgproc.COLOR_RGB2GRAY);
                        detector.detect(rgba, keyPoints);
                        Mat descriptors1 = new Mat();
                        extractor.compute(rgba, keyPoints, descriptors1);
                        if (l.equals("v")) {
                            vComp = "1";
                        } else {
                            vComp = "";
                        }
                        Mat m = keypointsToVector(keyPoints, descriptors1);
                        int col = m.cols();
                        int row = m.rows();
                        double Data[] = new double[m.cols()];
                        for (int i = 0; i < col; i++) {
                            double Dt[] = m.get(0, i);
                            Data[i] = Dt[0];
                            //Log.d(TAG, i+ "");
                        }
                        Log.d(TAG, "aqui");
                        trainData.put(index, 0, Data);
                        Log.d(TAG, "aqui2");
                        labelArray[index] = Integer.parseInt(v + vComp);
                        index++;
                    }
                }
            }
         //   Log.d(TAG, "Ate onde "+ aux);
        //}


        GeraSvmPoly(trainData,responses,labelArray);
        GeraSvmlinear(trainData,responses,labelArray);
        GeraSvmB(trainData,responses,labelArray);
    }


    private void GeraSVMSurf()
    {
        int rowsSize = 240;
        int[] labelArray = new int[rowsSize];
        int index = 0 ;
        int columnSize = (128000);
        Mat trainData = new Mat(rowsSize, columnSize, CvType.CV_32F);
        Mat responses = new Mat(rowsSize, 1, CvType.CV_32S);
        String vComp ="";
        StringBuffer textoBuffer = new StringBuffer("");

        SURF detector = SURF.create();
        detector.setHessianThreshold(300);
        detector.setNOctaveLayers(6);
        detector.setNOctaves(3);
        detector.setUpright(false);
        Mat rgba = new Mat();
        final String[] VALOR = { "2", "5", "10", "20", "50", "100" };
        final String[] LADO = { "f", "v" };
        // for (int aux = 1 ; aux < 13 ; aux ++ ) {
        for (int n = 11; n < 31; n++) {
            //for (int n = 1; n < 2; n++) {
            for (String v : VALOR) {
                for (String l : LADO) {
                    String vNomeImagem;
                    vNomeImagem = "s" + v + "_" + l + "" + n;
                    Log.d(TAG, vNomeImagem);
                    int drawable = getResources().getIdentifier(vNomeImagem, "drawable", getPackageName());
                    inputImage = BitmapFactory.decodeResource(getResources(), drawable);
                    Utils.bitmapToMat(inputImage, rgba);
                    MatOfKeyPoint keyPoints = new MatOfKeyPoint();
                    Imgproc.cvtColor(rgba, rgba, Imgproc.COLOR_RGB2GRAY);
                    detector.detect(rgba, keyPoints);
                    Mat descriptors1 = new Mat();
                    detector.compute(rgba, keyPoints, descriptors1);
                    if (l.equals("v")) {
                        vComp = "1";
                    } else {
                        vComp = "";
                    }
                    Mat m = keypointsToVectorSurf(keyPoints, descriptors1);
                    int col = m.cols();
                    int row = m.rows();
                    double Data[] = new double[m.cols()];
                    for (int i = 0; i < col; i++) {
                        double Dt[] = m.get(0, i);
                        Data[i] = Dt[0];
                        //Log.d(TAG, i+ "");
                    }
                    Log.d(TAG, "aqui");
                    trainData.put(index, 0, Data);
                    Log.d(TAG, "aqui2");
                    labelArray[index] = Integer.parseInt(v + vComp);
                    index++;
                }
            }
        }
        //   Log.d(TAG, "Ate onde "+ aux);
        //}


        GeraSvmPoly(trainData,responses,labelArray);
        GeraSvmlinear(trainData,responses,labelArray);
        GeraSvmB(trainData,responses,labelArray);
    }

    private  void  GeraSvmB(Mat trainData2, Mat responses2,int[] labelArray2)
    {
        svmBRF = SVM.create();
        svmBRF.setType(SVM.C_SVC);
        svmBRF.setKernel(SVM.RBF);
        //svmBRF.setC(1.0);
        //svmBRF.setDegree(1.0);
        //svmBRF.setCoef0(0.0);
        //svmBRF.setGamma(0.01);
        //svmBRF.setP(0.001);
       // svmBRF.setNu(0.001);
       // svmBRF.setTermCriteria(new TermCriteria(TermCriteria.EPS, 10000, 1e-12));
        responses2.put(0, 0, labelArray2);
        labelArray2 = null;
        Log.w(TAG, "Size of trainData: rows = " + trainData2.rows()
                + ", cols = " + trainData2.cols());
        Log.w(TAG, "Size of responses: rows = " + responses2.rows()
                + ", cols = " + responses2.cols());

        Log.i(TAG, "Treinando...");
        ParamGrid C = ParamGrid.create();
        ParamGrid p = ParamGrid.create();
        ParamGrid nu =ParamGrid.create();
        ParamGrid gamma = ParamGrid.create();
        gamma.set_logStep(0.0); // gamma fixo
        ParamGrid coeff = ParamGrid.create();
        ParamGrid degree = ParamGrid.create();
        int kFolds = 10 ;
        svmBRF.trainAuto(trainData2,ROW_SAMPLE,responses2,kFolds,C,gamma,p,nu,coeff,degree,false);
        if(svmBRF.isTrained()) {
            String vSql ="";
            Log.i(TAG, "Concluido!");
        }
        File datasetFile = new File(getExternalFilesDir(null).getAbsolutePath(),
                "svm_trainedb.xml");
        svmBRF.save(datasetFile.getAbsolutePath());
        Log.i(TAG, "SVM SALVA");
    }

    private  void  GeraSvmlinear(Mat trainData2, Mat responses2,int[] labelArray2)
    {
        svmLinear = SVM.create();
        svmLinear.setType(SVM.C_SVC);
        svmLinear.setKernel(SVM.LINEAR);
        //svmLinear.setC(1.0);
        //svmLinear.setDegree(1.0);
        //svmLinear.setCoef0(0.0);
        //svmLinear.setGamma(0.01);
       // svmLinear.setP(0.001);
        //svmLinear.setNu(0.001);
        //svmLinear.setTermCriteria(new TermCriteria(TermCriteria.EPS, 10000, 1e-12));
        responses2.put(0, 0, labelArray2);
        labelArray2 = null;
        Log.w(TAG, "Size of trainData: rows = " + trainData2.rows()
                + ", cols = " + trainData2.cols());
        Log.w(TAG, "Size of responses: rows = " + responses2.rows()
                + ", cols = " + responses2.cols());

        Log.i(TAG, "Treinando...");
        ParamGrid C = ParamGrid.create();
        ParamGrid p = ParamGrid.create();
        ParamGrid nu =ParamGrid.create();
        ParamGrid gamma = ParamGrid.create();
        gamma.set_logStep(0.0); // gamma fixo
        ParamGrid coeff = ParamGrid.create();
        ParamGrid degree = ParamGrid.create();
        int kFolds = 10 ;
        //svmLinear.trainAuto(trainData2,ROW_SAMPLE,responses2,kFolds,C,gamma,p,nu,coeff,degree,false);
        svmLinear.trainAuto(trainData2,ROW_SAMPLE,responses2);
        if(svmLinear.isTrained()) {
            String vSql ="";
            Log.i(TAG, "Concluido!");
        }
        File datasetFile = new File(getExternalFilesDir(null).getAbsolutePath(),
                "svm_trainedl.xml");
        svmLinear.save(datasetFile.getAbsolutePath());
        Log.i(TAG, "SVM SALVA");
    }



    private  void  GeraSvmPoly(Mat trainData2, Mat responses2,int[] labelArray2)
    {
        svmpoly = SVM.create();
        svmpoly.setType(SVM.NU_SVC);
        svmpoly.setKernel(SVM.POLY);
        svmpoly.setC(1.0);
        svmpoly.setDegree(1.0);
        svmpoly.setCoef0(0.0);
        svmpoly.setGamma(0.01);
        svmpoly.setP(0.001);
        svmpoly.setNu(0.001);
        svmpoly.setTermCriteria(new TermCriteria(TermCriteria.EPS, 10000, 1e-12));
        responses2.put(0, 0, labelArray2);
        labelArray2 = null;
        Log.w(TAG, "Size of trainData: rows = " + trainData2.rows()
                + ", cols = " + trainData2.cols());
        Log.w(TAG, "Size of responses: rows = " + responses2.rows()
                + ", cols = " + responses2.cols());

        Log.i(TAG, "Treinando...");
        ParamGrid C = ParamGrid.create();
        ParamGrid p = ParamGrid.create();
        ParamGrid nu =ParamGrid.create();
        ParamGrid gamma = ParamGrid.create();
        gamma.set_logStep(0.0); // gamma fixo
        ParamGrid coeff = ParamGrid.create();
        ParamGrid degree = ParamGrid.create();
        int kFolds = 10 ;
        svmpoly.trainAuto(trainData2,ROW_SAMPLE,responses2,kFolds,C,gamma,p,nu,coeff,degree,false);
        if(svmpoly.isTrained()) {
            String vSql ="";
            Log.i(TAG, "Concluido!");
        }
        File datasetFile = new File(getExternalFilesDir(null).getAbsolutePath(),
                "svm_trained.xml");
        svmpoly.save(datasetFile.getAbsolutePath());
        Log.i(TAG, "SVM SALVA");
    }





    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    try {
                        sift();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public Mat keypointsToVector(MatOfKeyPoint mat, Mat descriptors){
        ArrayList<Pontos> lista = new ArrayList<Pontos>();
        if(mat!=null && !mat.empty()){
            KeyPoint[] array = mat.toArray();
            Mat descriptorsFoto = descriptors;
            for (int i = 0; i < array.length; i++) {
                double[] desc = new double[descriptorsFoto.cols()];
                for (int j = 0; j < descriptorsFoto.cols(); j++) { // 64
                    desc[j] = descriptorsFoto.get(i, j)[0];
                }
                lista.add(new Pontos(array[i].response, desc));
            }
        }

        Collections.sort(lista);
        List<Mat> matDesc = new ArrayList<Mat>();

        int cont = 0;
        int ind = (lista.size() - 1); // comecar do ultimo (maior)
        while (cont < 178) {
            // Matriz auxiliar
            //Mat m = new Mat(1, 64, CvType.CV_32F);
            //for (int i = 0; i < 64; i++) {
            //    m.put(0, i, lista.get(ind).getDescriptors()[i]);
            //}
            Mat m = new Mat(1, 32, CvType.CV_32F);
            for (int i = 0; i < 32; i++) {
                m.put(0, i, lista.get(ind).getDescriptors()[i]);
            }
            // Adicionar a lista
            matDesc.add(m);

            ind--;
            cont++;
        }

        // Concatenar matrizes da lista horizontalmente (1D matriz)
        Mat data = new Mat();

        Core.hconcat(matDesc, data);
        Log.w(TAG, "Size of trainData: rows = " + data.rows()
                + ", cols = " + data.cols());
        return data;

    }

    public Mat keypointsToVectorSurf(MatOfKeyPoint mat, Mat descriptors){
        ArrayList<Pontos> lista = new ArrayList<Pontos>();
        if(mat!=null && !mat.empty()){
            KeyPoint[] array = mat.toArray();
            Mat descriptorsFoto = descriptors;
            for (int i = 0; i < array.length; i++) {
                double[] desc = new double[descriptorsFoto.cols()];
                for (int j = 0; j < descriptorsFoto.cols(); j++) { // 64
                    desc[j] = descriptorsFoto.get(i, j)[0];
                }
                lista.add(new Pontos(array[i].response, desc));
            }
        }

        Collections.sort(lista);
        List<Mat> matDesc = new ArrayList<Mat>();

        int cont = 0;
        int ind = (lista.size() - 1); // comecar do ultimo (maior)

        while (cont < 2000) {
            // Matriz auxiliar
            //Mat m = new Mat(1, 64, CvType.CV_32F);
            //for (int i = 0; i < 64; i++) {
            //    m.put(0, i, lista.get(ind).getDescriptors()[i]);
            //}
            Mat m = new Mat(1, 64, CvType.CV_32F);
            for (int i = 0; i < 64; i++) {
                m.put(0, i, lista.get(ind).getDescriptors()[i]);
            }
            // Adicionar a lista
            matDesc.add(m);

            ind--;
            cont++;
        }

        // Concatenar matrizes da lista horizontalmente (1D matriz)
        Mat data = new Mat();

        Core.hconcat(matDesc, data);
        Log.w(TAG, "Size of trainData: rows = " + data.rows()
                + ", cols = " + data.cols());
        return data;

    }

    public void saveResults(String t, StringBuffer txt){
        String filename = getExternalFilesDir(null).getAbsolutePath() + "/"+t+".txt";
        try {
            File new_file = new File(filename);
            FileOutputStream fos = new FileOutputStream(new_file);
            fos.write(txt.toString().getBytes());
            fos.flush();
            fos.close();
            Log.i("APP", "==> Resultados salvos");

        } catch (Exception e) {
            Log.e("APP", "Não achou o arquivo: " + e.getMessage());
        }
    }







}