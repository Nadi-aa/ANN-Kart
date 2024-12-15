using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

// This script implements an Artificial Neural Network (ANN) for controlling a Unity object.
// The ANN is trained using pre-recorded data or loaded from a saved state.
public class ANNDrive : MonoBehaviour
{
    // ANN instance to process input and generate output
    ANN ann;

    // Visible distance for raycasting in meters
    public float visibleDistance = 200.0f;

    // Number of training iterations
    public int epochs = 50000;

    // Movement parameters for the object
    public float speed = 50.0F;
    public float rotationSpeed = 100.0F;

    // State variables for training
    bool trainingDone = false;
    float trainingProgress = 0;
    double sse = 0; // Sum of Squared Errors
    double lastSSE = 1;

    // Stores the current translation and rotation
    public float translation;
    public float rotation;

    // Flag to load weights from a file instead of training from scratch
    public bool loadFromFile = false;

    // Initialization
    void Start()
    {
        // Initialize the ANN with specific parameters: input nodes, output nodes, hidden layers, neurons per layer, and learning rate
        ann = new ANN(5, 2, 1, 10, 0.00005);

        // Load pre-trained weights or start training
        if (loadFromFile)
        {
            LoadWeightsFromFile();
            trainingDone = true;
        }
        else
            StartCoroutine(LoadTrainingSet());
    }

    // Display training progress and parameters on the GUI
    void OnGUI()
    {
        GUI.Label(new Rect(25, 25, 250, 30), "SSE: " + lastSSE);
        GUI.Label(new Rect(25, 40, 250, 30), "Alpha: " + ann.alpha);
        GUI.Label(new Rect(25, 55, 250, 30), "Trained: " + trainingProgress);
    }

    // Coroutine to load the training dataset and train the ANN
    IEnumerator LoadTrainingSet()
    {
        string path = Application.dataPath + "/trainingData2.txt";
        string line;

        if (File.Exists(path))
        {
            // Count the number of lines in the training dataset
            int lineCount = File.ReadAllLines(path).Length;

            // Open the dataset file for reading
            StreamReader tdf = File.OpenText(path);

            // Lists to store inputs and outputs for the ANN
            List<double> calcOutputs = new List<double>();
            List<double> inputs = new List<double>();
            List<double> outputs = new List<double>();

            for (int i = 0; i < epochs; i++)
            {
                sse = 0;
                tdf.BaseStream.Position = 0; // Reset file pointer to the beginning

                // Save the current weights for possible rollback
                string currentWeights = ann.PrintWeights();

                while ((line = tdf.ReadLine()) != null)
                {
                    string[] data = line.Split(',');

                    // Ignore lines with zero translation and rotation values
                    if (System.Convert.ToDouble(data[5]) != 0 && System.Convert.ToDouble(data[6]) != 0)
                    {
                        inputs.Clear();
                        outputs.Clear();

                        // Read inputs (sensor data from raycasting)
                        inputs.Add(System.Convert.ToDouble(data[0]));
                        inputs.Add(System.Convert.ToDouble(data[1]));
                        inputs.Add(System.Convert.ToDouble(data[2]));
                        inputs.Add(System.Convert.ToDouble(data[3]));
                        inputs.Add(System.Convert.ToDouble(data[4]));

                        // Normalize outputs (translation and rotation)
                        outputs.Add(Map(0, 1, -1, 1, System.Convert.ToSingle(data[5])));
                        outputs.Add(Map(0, 1, -1, 1, System.Convert.ToSingle(data[6])));

                        // Train the ANN and calculate the error
                        calcOutputs = ann.Train(inputs, outputs);
                        float thisError = ((Mathf.Pow((float)(outputs[0] - calcOutputs[0]), 2) +
                                            Mathf.Pow((float)(outputs[1] - calcOutputs[1]), 2))) / 2.0f;
                        sse += thisError;
                    }
                }

                // Update training progress and average error
                trainingProgress = (float)i / epochs;
                sse /= lineCount;

                // Adjust learning rate and reload weights if necessary
                if (lastSSE < sse)
                {
                    ann.LoadWeights(currentWeights);
                    ann.alpha = Mathf.Clamp((float)ann.alpha - 0.001f, 0.01f, 0.9f);
                }
                else
                {
                    ann.alpha = Mathf.Clamp((float)ann.alpha + 0.001f, 0.01f, 0.9f);
                    lastSSE = sse;
                }

                yield return null; // Wait for the next frame
            }
        }

        trainingDone = true;

        // Save the trained weights for future use
        if (!loadFromFile)
            SaveWeightsToFile();
    }

    // Save ANN weights to a file
    void SaveWeightsToFile()
    {
        string path = Application.dataPath + "/weights.txt";
        StreamWriter wf = File.CreateText(path);
        wf.WriteLine(ann.PrintWeights());
        wf.Close();
    }

    // Load ANN weights from a file
    void LoadWeightsFromFile()
    {
        string path = Application.dataPath + "/weights.txt";
        if (File.Exists(path))
        {
            StreamReader wf = File.OpenText(path);
            string line = wf.ReadLine();
            ann.LoadWeights(line);
        }
    }

    // Normalize a value from one range to another
    float Map(float newfrom, float newto, float origfrom, float origto, float value)
    {
        if (value <= origfrom)
            return newfrom;
        else if (value >= origto)
            return newto;
        return (newto - newfrom) * ((value - origfrom) / (origto - origfrom)) + newfrom;
    }

    // Round a float to the nearest 0.5 increment
    float Round(float x)
    {
        return (float)System.Math.Round(x, System.MidpointRounding.AwayFromZero) / 2.0f;
    }

    // Update method to process ANN output and control object movement
    void Update()
    {
        if (!trainingDone) return;

        List<double> calcOutputs = new List<double>();
        List<double> inputs = new List<double>();

        // Raycasting to detect obstacles and gather sensor data
        RaycastHit hit;
        float fDist = 0, rDist = 0, lDist = 0, r45Dist = 0, l45Dist = 0;

        if (Physics.Raycast(transform.position, this.transform.forward, out hit, visibleDistance))
            fDist = 1 - Round(hit.distance / visibleDistance);

        if (Physics.Raycast(transform.position, this.transform.right, out hit, visibleDistance))
            rDist = 1 - Round(hit.distance / visibleDistance);

        if (Physics.Raycast(transform.position, -this.transform.right, out hit, visibleDistance))
            lDist = 1 - Round(hit.distance / visibleDistance);

        if (Physics.Raycast(transform.position,
                            Quaternion.AngleAxis(-45, Vector3.up) * this.transform.right,
                            out hit, visibleDistance))
            r45Dist = 1 - Round(hit.distance / visibleDistance);

        if (Physics.Raycast(transform.position,
                            Quaternion.AngleAxis(45, Vector3.up) * -this.transform.right,
                            out hit, visibleDistance))
            l45Dist = 1 - Round(hit.distance / visibleDistance);

        // Pass sensor data to the ANN
        inputs.Add(fDist);
        inputs.Add(rDist);
        inputs.Add(lDist);
        inputs.Add(r45Dist);
        inputs.Add(l45Dist);

        // Calculate outputs (translation and rotation)
        calcOutputs = ann.CalcOutput(inputs, new List<double> { 0, 0 });

        // Map ANN outputs back to -1 to 1 range
        float translationInput = Map(1, -1, 0, 1, (float)calcOutputs[0]);
        float rotationInput = Map(-1, 1, 0, 1, (float)calcOutputs[1]);

        // Apply movement to the object
        translation = translationInput * speed * Time.deltaTime;
        rotation = rotationInput * rotationSpeed * Time.deltaTime;
        this.transform.Translate(0, 0, translation);
        this.transform.Rotate(0, rotation, 0);
    }
}