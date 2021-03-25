//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!!
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Szabó Bence Sándor
// Neptun : NQB6UO
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

    uniform vec3 m1;
    uniform vec3 m2;
    uniform float d;

	layout(location = 0) in vec3 vp;
    layout(location = 1) in vec2 uv;

    out vec2 textCoord;

    float Lorentz(vec3 q, vec3 p) {
        return dot(q.xy, p.xy) - q.z * p.z;
    }

    float distance(vec3 p, vec3 q) {
        return acosh(-Lorentz(q, p));
    }

    vec3 mirror(vec3 p, vec3 m) {
        float d = distance(p, m);
        vec3 v = (m - p * cosh(d));
        return p * cosh(2.0f * d) + v * 2 * cosh(d);
    }

	void main() {
        vec3 mirroredVp;
        if (sinh(d) == 0 || d < 0.01)
            mirroredVp = vp;
        else
            mirroredVp = mirror(mirror(vp, m1), m2);

        textCoord = uv;
		gl_Position = vec4(mirroredVp.x, mirroredVp.y, 0, mirroredVp.z);		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentNode = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers

    uniform sampler2D textureUnit;
    in vec2 textCoord;

	out vec4 outColor;		// computed color of the current pixel

	void main() {
        outColor = texture(textureUnit, textCoord);	// computed color is the color of the primitive
	}
)";

// fragment shader in GLSL
const char * const fragmentEdge = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers

	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
        outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

bool mouseDown = false;

bool operator!=(vec3 a, vec3 b) {
    return (a.x != b.x && a.y != b.y && a.z != b.z);
}

bool operator==(vec3 a, vec3 b) {
    return (a.x == b.x && a.y == b.y && a.z == b.z);
}

bool operator==(vec4 a, vec4 b) {
    return (a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w);
}

vec3 toHyperbola(vec2 pos) {
    if (pos == vec2(0, 0))
        return vec3(0, 0, 1);
    float d = sqrtf(powf(pos.x, 2) + powf(pos.y, 2));
    vec2 v = {pos.x / d, pos.y / d};
    return {v.x * sinhf(d), v.y * sinhf(d), coshf(d)};
}

vec3 invHyper(vec2 pos) {
    float s = sqrt(1.0f - dot(pos, pos));
    return vec3(pos.x, pos.y, 1) / s;
}

float Lorentz(vec3 q, vec3 p) {
    return dot(vec2(q.x, q.y), vec2(p.x, p.y)) - q.z * p.z;
}

float distance(vec3 p, vec3 q) {
    return acosh(-Lorentz(q, p));
}

vec3 mirror(vec3 p, vec3 m) {
    float d = distance(p, m);
    vec3 v = (m - p * coshf(d));  // /sinh(d)
    return p * coshf(2.0f * d) + v * 2 * coshf(d); //sinh(2d) = 2 * sinh(d) * cosh(d)
}

vec3 hyperTrans(vec3 hyperPos, vec3 from, vec3 to) {
    float dTmp = distance(from, to);
    if (sinhf(dTmp) != 0) {
        vec3 vTmp = (to - from * coshf(dTmp)) / sinhf(dTmp);
        vec3 m1Tmp = from * coshf(dTmp / 4.0f) + vTmp * sinh(dTmp / 4.0f);
        vec3 m2Tmp = from * coshf((3.0f * dTmp) / 4.0f) + vTmp * sinhf((3.0f * dTmp) / 4.0f);
        return mirror(mirror(hyperPos, m1Tmp), m2Tmp);
    } else {
        return hyperPos;
    }
}

vec3 hyperTrans(vec3 hyperPos, vec3 v) {
    float dTmp = length(v);
    vec3 vTmp = normalize(v);
    return hyperPos * coshf(dTmp) + vTmp * sinhf(dTmp);
}

//Globals:
GPUProgram gpuProgramNode, gpuProgramEdge;
int nv = 100;
vec3 from = {0, 0, -1}, to = {0, 0, -1};
vec3 m1, m2;
vec3 v;
float d;
bool sortStarted = false;
float startedSec;

//parameters:
float prefDist = 0.2f;
float fParam = 5.0f;
float hParam = 1.25;
float nodeMass = 1.0f;
float q = 20.0;

class Node {
    unsigned int vaoCircle{}, vboCircle[2]{};
    vec3 hPos;
    vec2 pos;
    vec3 color[2];
    Texture * texture{};
    vec3* vertices;
    vec2* uvVertices;
    std::vector<Node*> neibours;
    vec3 vi;

    void generateTexture(vec3 c1, vec3 c2) {
        std::vector<vec4> textureVec;

        for (int i = 0; i < 50; ++i)
            for (int j = 0; j < 50; ++j) {
                float x = (float)i / 50.0f;
                float y = (float)j / 50.0f;
                if (powf(x - 0.5, 2) + powf(y - 0.5, 2) <= 0.1)
                    textureVec.emplace_back(vec4(c2.x, c2.y, c2.z, 1));
                else
                    textureVec.emplace_back(vec4(c1.x, c1.y, c1.z, 1));

                /*if (abs(x - 0.5f) < 0.1f or abs(y - 0.5f) < 0.1)
                    textureVec.emplace_back(vec4(c2.x, c2.y, c2.z, 1));
                else
                    textureVec.emplace_back(vec4(c1.x, c1.y, c1.z, 1));*/

            }
        texture = new Texture(50, 50, textureVec);
    }

    float f(float dis) {
        return fParam * powf(dis - prefDist, 3);
    }

    float h(float dis) {
        if (dis < 0.001)
            dis += 0.01;

        return - (1 / (hParam * dis));
        //return dis < 3.0f ? - (1 / powf(M_E, hParam * dis - prefDist)) : 0;
    }

public:
    Node(vec2 pos, vec3 color1, vec3 color2) : hPos(toHyperbola(pos)), pos(pos), vi(0, 0, 0) {
        color[0] = color1;
        color[1] = color2;

        vertices = new vec3[nv];
        uvVertices = new vec2[nv];

        for (int i = 0; i < nv; ++i) {
            float fi = i * 2 * M_PI / nv;
            vertices[i] = toHyperbola(vec2(0.045f * cosf(fi), 0.045f * sinf(fi)));
            uvVertices[i] = vec2(0.5f + 0.5f * cosf(fi), 0.5f + 0.5f * sinf(fi));
        }

        vec3 nullPoint = vec3(0, 0, 1);

        for (int i = 0; i < nv; ++i)
            vertices[i] = hyperTrans(vertices[i], nullPoint, hPos);

        generateTexture(color1, color2);

        //Make Nodes vaoCircle/vbo
        gpuProgramNode.Use();
        glGenVertexArrays(1, &vaoCircle);
        glBindVertexArray(vaoCircle);

        glGenBuffers(2, vboCircle);
        glBindBuffer(GL_ARRAY_BUFFER, vboCircle[0]);

        glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * nv, vertices, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);  // AttribArray 0
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

        glBindBuffer(GL_ARRAY_BUFFER, vboCircle[1]);

        glBufferData(GL_ARRAY_BUFFER, sizeof(vec2) * nv, uvVertices, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(1);  // AttribArray 0
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    }

    vec2 get2d() {
        return pos;
    }

    vec3 hyperPos() {
        return hPos;
    }

    void setPostByEuk(vec2 pos) {
        this->pos = pos;
        this->hPos = toHyperbola(pos);
    }

    void setHyper(vec3 pos) {
        hPos = pos;
        hPos.z = sqrtf(1 + hPos.x * hPos.x + hPos.y * hPos.y);
    }

    bool different(vec3 c1, vec3 c2) {
        return color[0] != c1 || color[0] != c2 || color[1] != c1 || color[1] != c2;
    }

    bool different(vec3 c1, vec3 c2, float avg) {
        return (length(c1 - color[0]) < avg && length(c2 - color[1]) < avg);
    }

    int getDiv() {
        return neibours.size();
    }

    void applyTrans() {
        if (sinh(d) != 0 and d > 0.01) {
            hPos = mirror(mirror(hPos, m1), m2);

            for (int i = 0; i < nv; ++i) {
                float fi = i * 2 * M_PI / nv;
                vertices[i] = toHyperbola({0.045f * cosf(fi), 0.045f * sinf(fi)});
            }

            vec3 nullPoint = vec3(0, 0, 1);
            for (int i = 0; i < nv; ++i)
                vertices[i] = hyperTrans(vertices[i], nullPoint, hPos);

            glBindVertexArray(vaoCircle);

            glBindBuffer(GL_ARRAY_BUFFER, vboCircle[0]);
            glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * nv, vertices, GL_DYNAMIC_DRAW);
        }
    }

    void Draw() {
        gpuProgramNode.setUniform(*texture, "textureUnit");
        glBindVertexArray(vaoCircle);
        glDrawArrays(GL_TRIANGLE_FAN, 0 , nv);
    }

    void addNei(Node* node) {
        bool contains = false;
        for (Node * nei : neibours)
            if (node == nei) {
                contains = true;
                break;
            }
        if (!contains)
            neibours.push_back(node);
    }

    float avgDisFromNeis() {
        if (!neibours.empty()) {
            float sumDis = 0;
            for (auto node : neibours) {
                sumDis += distance(hyperPos(), node->hyperPos());
            }
            return sumDis / (float) neibours.size();
        } else {
            return 0;
        }
    }

    void refresh() {
        for (int i = 0; i < nv; ++i) {
            float fi = i * 2 * M_PI / nv;
            vertices[i] = toHyperbola({0.045f * cosf(fi), 0.045f * sinf(fi)});
        }

        vec3 nullPoint = vec3(0, 0, 1);
        for (int i = 0; i < nv; ++i)
            vertices[i] = hyperTrans(vertices[i], nullPoint, hPos);

        gpuProgramNode.Use();
        glBindVertexArray(vaoCircle);

        glBindBuffer(GL_ARRAY_BUFFER, vboCircle[0]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * nv, vertices, GL_DYNAMIC_DRAW);
    }

    vec3 force(Node* node) {
        bool nei = false;
        for (Node* neiNode : neibours) {
            if (neiNode == node) {
                nei = true;
                break;
            }
        }

        float dis = distance(node->hyperPos(),hPos);

        vec3 ret = normalize(node->hyperPos() - hPos * coshf(dis) / sinhf(dis));

        ret = ret * (nei ? f(dis) : h(dis));

        return ret;
    }

    vec3 force(vec3 v) {
        bool nei = false;

        float dis = distance(v,hPos);

        vec3 ret = normalize(v - hPos * coshf(dis) / sinhf(dis));

        return ret;
    }

    vec3& getV() {
        return vi;
    }

    void setV(vec3 v) {
        vi = v;
    }

    ~Node() {
        delete[] vertices;
        delete[] uvVertices;
    }
};

void swapPos(Node* n1, Node* n2) {
    vec3 tmp = n1->hyperPos();
    n1->setHyper(n2->hyperPos());
    n2->setHyper(tmp);
}

class Edge {
    Node * n1;
    Node * n2;
    unsigned int vao, vbo;
    vec3 points[2];

public:

    Edge(Node * n1, Node * n2)
            : n1(n1), n2(n2) {
        //Make Nodes vaoCircle/vbo
        points[0] = n1->hyperPos();
        points[1] = n2->hyperPos();

        //Make Nodes vaoCircle/vbo
        gpuProgramEdge.Use();

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glEnableVertexAttribArray(0);  // AttribArray 0
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * 2, points, GL_DYNAMIC_DRAW);
    }

    void refresh() {
        points[0] = n1->hyperPos();
        points[1] = n2->hyperPos();

        //Make Nodes vaoCircle/vbo
        gpuProgramEdge.Use();

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * 2, points, GL_DYNAMIC_DRAW);
    }

    void draw() {
        glBindVertexArray(vao);
        glDrawArrays(GL_LINES, 0 , 2);
    }

    void applyTrans() {
        refresh();
    }
};

class Graph {
    std::vector<Node*> nodes;
    std::vector<Edge*> edges;

    void generateNodes() {
        while (nodes.size() < 50) {
            float randX = ((float)(rand() % 200 - 100) / 100.0f);
            float randY = ((float)(rand() % 200 - 100) / 100.0f);

            bool isNear = false;

            for (auto node : nodes) {
                vec2 tmp = node->get2d();
                if (length(vec2(randX, randY) - tmp) < 0.1) {
                    isNear = true;
                    continue;
                }
            }
            vec3 color1;
            vec3 color2;
            bool identical = true;
            while (identical) {
                identical = false;
                color1 = vec3((float) (rand() % 80) / 100.0f + 0.2, (float) (rand() % 80) / 100.0f + 0.2,
                                   (float) (rand() % 80) / 100.0f + 0.2);
                color2 = vec3((float) (rand() % 80) / 100.0f + 0.2, (float) (rand() % 80) / 100.0f + 0.2,
                                   (float) (rand() % 80) / 100.0f + 0.2);
                for (auto node : nodes) {
                    if (!node->different(color1, color2) || node->different(color1, color2, 0.3))
                        identical = true;
                }
            }
            if (!isNear && color1 != color2 && length(color2 - color1) > 0.9)
                nodes.push_back(new Node(vec2(randX, randY), color1, color2));
        }
    }

    void generateEdges() {
        int maxEdgenum = (50 * 49) / 2;
        std::vector<vec2> pairs;
        while (edges.size() < 63) {
            int n1 = rand() % 50;
            int n2 = rand() % 50;

            if (n1 != n2) {
                bool has = false;
                for (auto & tmp : pairs) {
                    if (tmp == vec2(n1, n2) || tmp == vec2(n2, n1)) {
                        has = true;
                        break;
                    }
                }
                if (!has) {
                    edges.push_back(new Edge(nodes[n1], nodes[n2]));
                    pairs.emplace_back(n1, n2);
                    nodes[n1]->addNei(nodes[n2]);
                    nodes[n2]->addNei(nodes[n1]);
                }
            }
        }
    }

    float pointGraphByAvgNei() {
        float sumAvgDis = 0;
        for (auto node : nodes) {
            sumAvgDis += node->avgDisFromNeis();
        }
        return sumAvgDis / (float)nodes.size();
    }

public:
    void Create() {
        //Generate Nodes location.
        generateNodes();

        //Generate edges.
        generateEdges();
    }

    void applyTrans() {
        gpuProgramNode.Use();
        for (Node * node : nodes)
            node->applyTrans();

        gpuProgramEdge.Use();
        for (Edge * edge : edges)
            edge->applyTrans();
    }

    void Draw() {
        //draw Edges
        gpuProgramEdge.Use();
       if (from == vec2(0, 0) || to == vec2(0, 0)) {
            gpuProgramEdge.setUniform(vec3(0, 0, 0), "m1");
            gpuProgramEdge.setUniform(vec3(0, 0, 0), "m2");
       } else {
            gpuProgramEdge.setUniform(m1, "m1");
            gpuProgramEdge.setUniform(m2, "m2");
       }
        gpuProgramEdge.setUniform(d, "d");
        gpuProgramEdge.setUniform(vec3(1,1,0), "color");
        glLineWidth(2.0f);

        std::size_t i = 0;
        for (Edge * edge : edges) {
            if (i != 0) {
                edge->draw();
            }
            ++i;
        }

        gpuProgramNode.Use();
        if (from == vec2(0, 0) || to == vec2(0, 0)) {
            gpuProgramNode.setUniform(vec3(0, 0, 0), "m1");
            gpuProgramNode.setUniform(vec3(0, 0, 0), "m2");
        } else {
            gpuProgramNode.setUniform(m1, "m1");
            gpuProgramNode.setUniform(m2, "m2");
        }
        gpuProgramNode.setUniform(d, "d");
        //draw Nodes
        for (Node * node : nodes)
            node->Draw();
    }
    
    void heuristic() {
        //sort nodes by div
        for (int i = 0; i < nodes.size(); ++i)
            for (int j = i; j < nodes.size();++j)
                if (nodes[i]->getDiv() < nodes[j]->getDiv())
                    std::swap(nodes[i], nodes[j]);

        //init vars
        int diffLayers = 2;
        int nodeInLayer = 10;
        int processedNodes = 0;
        float rDiff = 0.4f;
        float layerR = rDiff;

        //place the center Node
        //nodes[processedNodes++]->setPostByEuk(vec2(0, 0));

        //place the others on layers.
        while (processedNodes < 50) {
            int tmpNum = std::min(nodeInLayer, 50 - processedNodes);

            for (int i = 0; i < tmpNum; ++i) {
                float phi = (i * 2.0f * M_PI) / tmpNum;
                nodes[processedNodes++]->setPostByEuk(vec2(layerR * cosf(phi), layerR * sinf(phi)));
            }

            nodeInLayer *= diffLayers;
            layerR += rDiff;
        }

        //correct nodes
        float tmpPoint = 0;
        while (tmpPoint != pointGraphByAvgNei()) {
            tmpPoint = pointGraphByAvgNei();
            int corrected = 0;
            int layer = 0;
            while (corrected < 50) {
                int tmpNodeInLayer = nodeInLayer + pow(diffLayers, layer);
                for (int i = corrected; i < 50; ++i) {
                    float bestDisDelta = 0;
                    int best = 0;
                    for (int j = i + 1; j < 50; ++j) {
                        float disAvgI = nodes[i]->avgDisFromNeis();
                        float disAvgJ = nodes[j]->avgDisFromNeis();

                        swapPos(nodes[i], nodes[j]);

                        float tmpDisAvgI = nodes[i]->avgDisFromNeis();
                        float tmpDisAvgJ = nodes[j]->avgDisFromNeis();

                        float disDelta = (tmpDisAvgI - disAvgI) + (tmpDisAvgJ - disAvgJ);
                        if (disDelta < bestDisDelta) {
                            bestDisDelta = disDelta;
                            best = j;
                        }

                        swapPos(nodes[i], nodes[j]);
                    }
                    if (bestDisDelta < 0)
                        swapPos(nodes[i], nodes[best]);

                    ++corrected;
                }
            }
        }

        //refresh nodes
        for (Node * node : nodes)
            node->refresh();

        //refresh edges
        for (Edge * edge : edges)
            edge->refresh();
    }

    void stepSort(float deltaTime) {

        for (Node* node : nodes) {
            vec3 forceSum(0,0,0);
            vec3 vi = node->getV();
            vec3 vTmp;
            vec3 ri = node->hyperPos();

            for (Node* other : nodes)
                if (node != other)
                    forceSum = forceSum + node->force(other);

            forceSum = forceSum + q * node->force(vec3(0, 0, 1));

            vTmp = vi * 0.03 + (forceSum * deltaTime / nodeMass);
            if (vTmp != 0) {
                node->setHyper(hyperTrans(ri, vTmp));

                float dis = length(vTmp * deltaTime);
                vTmp = length(vTmp) * (normalize(vTmp) * coshf(dis) + ri * sinhf(dis));

                node->setV(vTmp);
            } else
                node->setV(vec3(0, 0, 0));
        }


        //refresh nodes
        for (Node * node : nodes)
            node->refresh();

        //refresh edges
        for (Edge * edge : edges)
            edge->refresh();
    }
};

Graph graph;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// create program for the GPU
	gpuProgramNode.create(vertexSource, fragmentNode, "outColor");
	gpuProgramEdge.create(vertexSource, fragmentEdge, "outColor");

	graph.Create();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	graph.Draw();

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == ' ') {
        graph.heuristic();
        sortStarted = true;
        startedSec = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
    }

	glutPostRedisplay();
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
    glutPostRedisplay();
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
    float cX = 2.0f * pX / windowWidth - 1;
    float cY = 1.0f - 2.0f * pY / windowHeight;
    if (mouseDown) {
        to = invHyper(vec2(cX, cY));
        //m1 and m2 calc
        d = distance(from, to);
        v = (to - from * coshf(d)) / sinhf(d);
        m1 = from * coshf(d / 4.0f) + v * sinhf(d / 4.0f);
        m2 = from * coshf((3.0f * d) / 4.0f) + v * sinhf((3.0f * d) / 4.0f);
	}
    glutPostRedisplay();
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
    float cX = 2.0f * pX / windowWidth - 1;
    float cY = 1.0f - 2.0f * pY / windowHeight;
    if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
        mouseDown = true;
        from = invHyper(vec2(cX, cY));
        to = invHyper(vec2(cX, cY));
	} else if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP) {
        mouseDown = false;
        to = invHyper(vec2(cX, cY));
        graph.applyTrans();
        d = 0;
	} else {
        d = 0;
    }

    glutPostRedisplay();
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME);
	if (sortStarted) {
	    for (int i = 0; i < 1000; ++i)
            graph.stepSort(0.001);

	    sortStarted = false;
    }
    glutPostRedisplay();
}
