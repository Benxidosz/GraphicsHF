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

    uniform vec3 from;
    uniform vec3 to;

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
        float d = distance(from, to);
        vec3 v = (to - from * cosh(d)) / sinh(d);
        vec3 m1 = from * cosh (d / 4.0f) + v * sinh(d / 4.0f);
        vec3 m2 = from * cosh ((3.0f * d) / 4.0f) + v * sinh((3.0f * d) / 4.0f);

        vec3 mirroredVp;
        if (d == 0)
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

//Globals:
GPUProgram gpuProgramNode, gpuProgramEdge;
int nv = 100;
vec3 sVert = {0, 0, 0}, eVert = {0, 0, 0};

bool mouseDown = false;

bool operator!=(vec3 a, vec3 b) {
    return (a.x != b.x and a.y != b.y and a.z != b.z);
}

bool operator==(vec3 a, vec3 b) {
    return (a.x == b.x and a.y == b.y and a.z == b.z);
}

bool operator==(vec4 a, vec4 b) {
    return (a.x == b.x and a.y == b.y and a.z == b.z and a.w == b.w);
}

vec3 toHyperbola(vec2 pos) {
    float d = sqrtf(powf(pos.x, 2) + powf(pos.y, 2));
    vec2 v = {pos.x / d, pos.y / d};
    return {v.x * sinhf(d), v.y * sinhf(d), coshf(d)};
}

vec3 invHyper(vec2 pos) {
    float s = sqrt(1.0f - dot(pos, pos));
    return vec3(pos.x, pos.y, 1) / s;
}



class Node {
    unsigned int vaoCircle{}, vboCircle[2]{};
    vec3 hPos;
    vec2 pos;
    vec3 color[2];
    Texture * texture;

    void generateTexture(vec3 c1, vec3 c2) {
        std::vector<vec4> textureVec;
        for (int i = 0; i < 50; ++i)
            for (int j = 0; j < 50; ++j) {
                float x = (float)i / 50.0f;
                float y = (float)j / 50.0f;
//                if (powf(x - 0.5, 2) + powf(y - 0.5, 2) <= 0.1)
//                    textureVec.emplace_back(vec4(c2.x, c2.y, c2.z, 1));
//                else
//                    textureVec.emplace_back(vec4(c1.x, c1.y, c1.z, 1));

                if (abs(x - 0.5f) < 0.1f or abs(y - 0.5f) < 0.1)
                    textureVec.emplace_back(vec4(c2.x, c2.y, c2.z, 1));
                else
                    textureVec.emplace_back(vec4(c1.x, c1.y, c1.z, 1));

            }
        texture = new Texture(50, 50, textureVec);
    }

public:
    Node(vec2 pos, vec3 color1, vec3 color2) : hPos(toHyperbola(pos)), pos(pos) {
        color[0] = color1;
        color[1] = color2;
        //Generate circle vertexes.
        vec3 vertices[nv];
        vec2 uvVertices[nv];

        for (int i = 0; i < nv; ++i) {
            float fi = i * 2 * M_PI / nv;
            vertices[i] = toHyperbola({pos.x + 0.06f * cosf(fi), pos.y + 0.06f * sinf(fi)});
            uvVertices[i] = vec2(0.5f + 0.5f * cosf(fi), 0.5f + 0.5f * sinf(fi));
        }

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

    bool different(vec3 c1, vec3 c2) {
        return color[0] != c1 || color[0] != c2 || color[1] != c1 || color[1] != c2;
    }

    bool different(vec3 c1, vec3 c2, float avg) {
        return (length(c1 - color[0]) < avg and length(c2 - color[1]) < avg);
    }

    void Draw() {
        gpuProgramNode.setUniform(*texture, "textureUnit");
        glBindVertexArray(vaoCircle);
        glBindVertexArray(vaoCircle);
        glDrawArrays(GL_TRIANGLE_FAN, 0 , nv);
    }
};

struct Edge {
    vec3 n1, n2;

    Edge(vec3 n1, vec3 n2)
    : n1(n1), n2(n2) { }
};

class Graph {
    unsigned int vaoEdges, vboEdges;
    std::vector<Node> nodes;
    std::vector<Edge> edges;

    void generateNodes() {
        while (nodes.size() < 50) {
            float randX = ((float)(rand() % 300 - 150) / 100.0f);
            float randY = ((float)(rand() % 300 - 150) / 100.0f);

            bool near = false;

            for (auto & node : nodes) {
                vec2 tmp = node.get2d();
                if (length(vec2(randX, randY) - tmp) < 0.15) {
                    near = true;
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
                for (auto & node : nodes) {
                    if (!node.different(color1, color2) or node.different(color1, color2, 0.3))
                        identical = true;
                }
            }
            if (!near and color1 != color2 and length(color2 - color1) > 0.9)
                nodes.emplace_back(Node({randX, randY}, color1, color2));
        }
    }

    void generateEdges() {
        int maxEdgenum = (50 * 49) / 2;
        std::vector<vec2> pairs;
        while ((float)edges.size() / (float)maxEdgenum < 0.05) {
            int n1 = rand() % 50;
            int n2 = rand() % 50;

            vec3 n1Pos = nodes[n1].hyperPos();
            vec3 n2Pos = nodes[n2].hyperPos();

            if (n1 != n2) {
                bool has = false;
                for (auto tmp : pairs) {
                    if (tmp == vec2(n1, n2) || tmp == vec2(n2, n1)) {
                        has = true;
                    }
                }
                if (!has) {

                    edges.emplace_back(Edge(n1Pos, n2Pos));
                    pairs.emplace_back(n1, n2);
                }
            }
        }
    }

public:
    void Create() {
        //Generate Nodes location.
        generateNodes();

        //Generate edges.
        generateEdges();

        //Make Edge vaoCircle/vbo
        gpuProgramEdge.Use();
        glGenVertexArrays(1, &vaoEdges);
        glBindVertexArray(vaoEdges);

        glGenBuffers(1, &vboEdges);
        glBindBuffer(GL_ARRAY_BUFFER, vboEdges);
        glBufferData(GL_ARRAY_BUFFER, sizeof(Edge) * edges.size(), &edges[0], GL_STATIC_DRAW);// we do not change later

        glEnableVertexAttribArray(0);  // AttribArray 0
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    }

    void Draw() {
        //draw Edges
        gpuProgramEdge.Use();
       if (sVert == vec2(0, 0) || eVert == vec2(0, 0)) {
            gpuProgramEdge.setUniform(vec3(0, 0, 0), "from");
            gpuProgramEdge.setUniform(vec3(0, 0, 0), "to");
       } else {
            gpuProgramEdge.setUniform(sVert, "from");
            gpuProgramEdge.setUniform(eVert, "to");
       }
        gpuProgramEdge.setUniform(vec3(1,1,0), "color");
        glLineWidth(2.0f);
        glBindVertexArray(vaoEdges);
        glDrawArrays(GL_LINES, 0 , edges.size() * 2);

        gpuProgramNode.Use();
        if (sVert == vec2(0, 0) || eVert == vec2(0, 0)) {
            gpuProgramNode.setUniform(vec3(0, 0, 0), "from");
            gpuProgramNode.setUniform(vec3(0, 0, 0), "to");
        } else {
            gpuProgramNode.setUniform(sVert, "from");
            gpuProgramNode.setUniform(eVert, "to");
        }
        //draw Nodes
        for (auto & node : nodes)
            node.Draw();
    }
};

Graph graph;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
    srand(time(0));

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
	glutPostRedisplay();
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
    float cX = 2.0f * pX / windowWidth - 1;
    float cY = 1.0f - 2.0f * pY / windowHeight;
    if (mouseDown) {
        eVert = invHyper(vec2(cX, cY));
	}
    glutPostRedisplay();
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
    float cX = 2.0f * pX / windowWidth - 1;
    float cY = 1.0f - 2.0f * pY / windowHeight;
    if (button = GLUT_RIGHT_BUTTON and state == GLUT_DOWN) {
        mouseDown = true;
        sVert = invHyper(vec2(cX, cY));
	} else if (button = GLUT_RIGHT_BUTTON and state == GLUT_UP) {
        mouseDown = false;
        eVert = invHyper(vec2(cX, cY));
	}
    glutPostRedisplay();
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
