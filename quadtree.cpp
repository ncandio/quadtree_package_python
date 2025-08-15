
#include <Python.h>
#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>
#include <array>
#include <optional>
#include <cassert>

namespace {

struct Point {
    double x, y;
    PyObject* data;
    
    Point(double x, double y, PyObject* data = nullptr) : x(x), y(y), data(data) {
        if (data) Py_INCREF(data);
    }
    
    ~Point() {
        if (data) Py_DECREF(data);
    }
    
    Point(const Point&) = delete;
    Point& operator=(const Point&) = delete;
    
    Point(Point&& other) noexcept : x(other.x), y(other.y), data(other.data) {
        other.data = nullptr;
    }
    
    Point& operator=(Point&& other) noexcept {
        if (this != &other) {
            if (data) Py_DECREF(data);
            x = other.x;
            y = other.y;
            data = other.data;
            other.data = nullptr;
        }
        return *this;
    }
    
    bool operator==(const Point& other) const noexcept {
        return std::abs(x - other.x) < 1e-9 && std::abs(y - other.y) < 1e-9;
    }
};

struct Rectangle {
    double x, y, width, height;
    
    constexpr Rectangle(double x, double y, double w, double h) noexcept 
        : x(x), y(y), width(w), height(h) {}
    
    constexpr bool contains(double px, double py) const noexcept {
        return px >= x && px < x + width && py >= y && py < y + height;
    }
    
    constexpr bool intersects(const Rectangle& other) const noexcept {
        return !(other.x >= x + width || other.x + other.width <= x ||
                 other.y >= y + height || other.y + other.height <= y);
    }
};

class QuadTree {
private:
    static constexpr size_t CAPACITY = 4;
    static constexpr size_t MAX_DEPTH = 10;
    
    Rectangle boundary_;
    std::vector<std::unique_ptr<Point>> points_;
    std::array<std::unique_ptr<QuadTree>, 4> children_;
    bool divided_;
    size_t depth_;
    
    enum class Quadrant { NW = 0, NE = 1, SW = 2, SE = 3 };
    
public:
    explicit QuadTree(const Rectangle& boundary, size_t depth = 0) 
        : boundary_(boundary), divided_(false), depth_(depth) {
        points_.reserve(CAPACITY);
    }
    
    bool insert(double x, double y, PyObject* data = nullptr) {
        if (!boundary_.contains(x, y)) {
            return false;
        }
        
        if (points_.size() < CAPACITY || depth_ >= MAX_DEPTH) {
            points_.emplace_back(std::make_unique<Point>(x, y, data));
            return true;
        }
        
        if (!divided_) {
            subdivide();
        }
        
        return insertIntoChild(x, y, data);
    }
    
    std::vector<Point*> query(const Rectangle& range) const {
        std::vector<Point*> found;
        found.reserve(CAPACITY * 4);
        queryImpl(range, found);
        return found;
    }
    
    std::vector<std::pair<Point*, Point*>> detectCollisions(double radius) const {
        std::vector<std::pair<Point*, Point*>> collisions;
        auto all_points = query(boundary_);
        
        const double radiusSquared = radius * radius;
        
        for (size_t i = 0; i < all_points.size(); ++i) {
            for (size_t j = i + 1; j < all_points.size(); ++j) {
                const double dx = all_points[i]->x - all_points[j]->x;
                const double dy = all_points[i]->y - all_points[j]->y;
                const double distanceSquared = dx * dx + dy * dy;
                
                if (distanceSquared < radiusSquared) {
                    collisions.emplace_back(all_points[i], all_points[j]);
                }
            }
        }
        
        return collisions;
    }
    
    bool contains(double x, double y) const {
        if (!boundary_.contains(x, y)) return false;
        
        for (const auto& point : points_) {
            if (std::abs(point->x - x) < 1e-9 && std::abs(point->y - y) < 1e-9) {
                return true;
            }
        }
        
        if (divided_) {
            for (const auto& child : children_) {
                if (child && child->contains(x, y)) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    std::vector<Point*> getAllPoints() const {
        return query(boundary_);
    }
    
    size_t size() const {
        size_t count = points_.size();
        if (divided_) {
            for (const auto& child : children_) {
                if (child) {
                    count += child->size();
                }
            }
        }
        return count;
    }
    
    size_t depth() const {
        if (!divided_) {
            return depth_;
        }
        
        size_t maxChildDepth = depth_;
        for (const auto& child : children_) {
            if (child) {
                maxChildDepth = std::max(maxChildDepth, child->depth());
            }
        }
        return maxChildDepth;
    }
    
    size_t subdivisions() const {
        if (!divided_) return 0;
        
        size_t count = 1;
        for (const auto& child : children_) {
            if (child) {
                count += child->subdivisions();
            }
        }
        return count;
    }
    
    bool empty() const noexcept {
        return points_.empty() && !divided_;
    }
    
    const Rectangle& boundary() const noexcept {
        return boundary_;
    }

private:
    void subdivide() {
        assert(!divided_);
        
        const double halfWidth = boundary_.width * 0.5;
        const double halfHeight = boundary_.height * 0.5;
        
        children_[static_cast<size_t>(Quadrant::NW)] = 
            std::make_unique<QuadTree>(Rectangle(boundary_.x, boundary_.y, halfWidth, halfHeight), depth_ + 1);
        children_[static_cast<size_t>(Quadrant::NE)] = 
            std::make_unique<QuadTree>(Rectangle(boundary_.x + halfWidth, boundary_.y, halfWidth, halfHeight), depth_ + 1);
        children_[static_cast<size_t>(Quadrant::SW)] = 
            std::make_unique<QuadTree>(Rectangle(boundary_.x, boundary_.y + halfHeight, halfWidth, halfHeight), depth_ + 1);
        children_[static_cast<size_t>(Quadrant::SE)] = 
            std::make_unique<QuadTree>(Rectangle(boundary_.x + halfWidth, boundary_.y + halfHeight, halfWidth, halfHeight), depth_ + 1);
        
        divided_ = true;
    }
    
    bool insertIntoChild(double x, double y, PyObject* data) {
        assert(divided_);
        
        for (auto& child : children_) {
            if (child && child->insert(x, y, data)) {
                return true;
            }
        }
        return false;
    }
    
    void queryImpl(const Rectangle& range, std::vector<Point*>& found) const {
        if (!boundary_.intersects(range)) {
            return;
        }
        
        for (const auto& point : points_) {
            if (range.contains(point->x, point->y)) {
                found.push_back(point.get());
            }
        }
        
        if (divided_) {
            for (const auto& child : children_) {
                if (child) {
                    child->queryImpl(range, found);
                }
            }
        }
    }
};

} // anonymous namespace

struct QuadTreeObject {
    PyObject_HEAD
    std::unique_ptr<QuadTree> qtree;
};

static PyObject* quadtree_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    auto* self = reinterpret_cast<QuadTreeObject*>(type->tp_alloc(type, 0));
    if (self) {
        new(&self->qtree) std::unique_ptr<QuadTree>();
    }
    return reinterpret_cast<PyObject*>(self);
}

static int quadtree_init(QuadTreeObject* self, PyObject* args, PyObject* kwds) {
    double x, y, width, height;
    
    if (!PyArg_ParseTuple(args, "dddd", &x, &y, &width, &height)) {
        return -1;
    }
    
    if (width <= 0.0 || height <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "Width and height must be positive");
        return -1;
    }
    
    try {
        self->qtree = std::make_unique<QuadTree>(Rectangle(x, y, width, height));
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return -1;
    }
    
    return 0;
}

static void quadtree_dealloc(QuadTreeObject* self) {
    self->qtree.~unique_ptr();
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

static PyObject* quadtree_insert(QuadTreeObject* self, PyObject* args) {
    double x, y;
    PyObject* data = nullptr;
    
    if (!PyArg_ParseTuple(args, "dd|O", &x, &y, &data)) {
        return nullptr;
    }
    
    if (!self->qtree) {
        PyErr_SetString(PyExc_RuntimeError, "QuadTree not initialized");
        return nullptr;
    }
    
    try {
        const bool success = self->qtree->insert(x, y, data);
        return PyBool_FromLong(success);
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    }
}

static PyObject* quadtree_contains(QuadTreeObject* self, PyObject* args) {
    double x, y;
    
    if (!PyArg_ParseTuple(args, "dd", &x, &y)) {
        return nullptr;
    }
    
    if (!self->qtree) {
        PyErr_SetString(PyExc_RuntimeError, "QuadTree not initialized");
        return nullptr;
    }
    
    return PyBool_FromLong(self->qtree->contains(x, y));
}

static PyObject* quadtree_get_all_points(QuadTreeObject* self, PyObject* Py_UNUSED(ignored)) {
    if (!self->qtree) {
        PyErr_SetString(PyExc_RuntimeError, "QuadTree not initialized");
        return nullptr;
    }
    
    try {
        auto results = self->qtree->getAllPoints();
        
        PyObject* result_list = PyList_New(results.size());
        if (!result_list) {
            return nullptr;
        }
        
        for (size_t i = 0; i < results.size(); ++i) {
            PyObject* tuple = PyTuple_New(results[i]->data ? 3 : 2);
            if (!tuple) {
                Py_DECREF(result_list);
                return nullptr;
            }
            
            PyTuple_SetItem(tuple, 0, PyFloat_FromDouble(results[i]->x));
            PyTuple_SetItem(tuple, 1, PyFloat_FromDouble(results[i]->y));
            if (results[i]->data) {
                Py_INCREF(results[i]->data);
                PyTuple_SetItem(tuple, 2, results[i]->data);
            }
            PyList_SetItem(result_list, i, tuple);
        }
        
        return result_list;
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    }
}

static PyObject* quadtree_subdivisions(QuadTreeObject* self, PyObject* Py_UNUSED(ignored)) {
    if (!self->qtree) {
        PyErr_SetString(PyExc_RuntimeError, "QuadTree not initialized");
        return nullptr;
    }
    
    return PyLong_FromSize_t(self->qtree->subdivisions());
}

static PyObject* quadtree_query(QuadTreeObject* self, PyObject* args) {
    double x, y, width, height;
    
    if (!PyArg_ParseTuple(args, "dddd", &x, &y, &width, &height)) {
        return nullptr;
    }
    
    if (!self->qtree) {
        PyErr_SetString(PyExc_RuntimeError, "QuadTree not initialized");
        return nullptr;
    }
    
    if (width < 0.0 || height < 0.0) {
        PyErr_SetString(PyExc_ValueError, "Width and height must be non-negative");
        return nullptr;
    }
    
    try {
        auto results = self->qtree->query(Rectangle(x, y, width, height));
        
        PyObject* result_list = PyList_New(results.size());
        if (!result_list) {
            return nullptr;
        }
        
        for (size_t i = 0; i < results.size(); ++i) {
            PyObject* tuple = PyTuple_New(results[i]->data ? 3 : 2);
            if (!tuple) {
                Py_DECREF(result_list);
                return nullptr;
            }
            
            PyTuple_SetItem(tuple, 0, PyFloat_FromDouble(results[i]->x));
            PyTuple_SetItem(tuple, 1, PyFloat_FromDouble(results[i]->y));
            if (results[i]->data) {
                Py_INCREF(results[i]->data);
                PyTuple_SetItem(tuple, 2, results[i]->data);
            }
            PyList_SetItem(result_list, i, tuple);
        }
        
        return result_list;
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    }
}

static PyObject* quadtree_detect_collisions(QuadTreeObject* self, PyObject* args) {
    double radius;
    
    if (!PyArg_ParseTuple(args, "d", &radius)) {
        return nullptr;
    }
    
    if (!self->qtree) {
        PyErr_SetString(PyExc_RuntimeError, "QuadTree not initialized");
        return nullptr;
    }
    
    if (radius < 0.0) {
        PyErr_SetString(PyExc_ValueError, "Radius must be non-negative");
        return nullptr;
    }
    
    try {
        auto collisions = self->qtree->detectCollisions(radius);
        
        PyObject* result_list = PyList_New(collisions.size());
        if (!result_list) {
            return nullptr;
        }
        
        for (size_t i = 0; i < collisions.size(); ++i) {
            PyObject* collision_dict = PyDict_New();
            if (!collision_dict) {
                Py_DECREF(result_list);
                return nullptr;
            }
            
            PyObject* p1_tuple = PyTuple_New(collisions[i].first->data ? 3 : 2);
            PyObject* p2_tuple = PyTuple_New(collisions[i].second->data ? 3 : 2);
            
            if (!p1_tuple || !p2_tuple) {
                Py_XDECREF(p1_tuple);
                Py_XDECREF(p2_tuple);
                Py_DECREF(collision_dict);
                Py_DECREF(result_list);
                return nullptr;
            }
            
            PyTuple_SetItem(p1_tuple, 0, PyFloat_FromDouble(collisions[i].first->x));
            PyTuple_SetItem(p1_tuple, 1, PyFloat_FromDouble(collisions[i].first->y));
            if (collisions[i].first->data) {
                Py_INCREF(collisions[i].first->data);
                PyTuple_SetItem(p1_tuple, 2, collisions[i].first->data);
            }
            
            PyTuple_SetItem(p2_tuple, 0, PyFloat_FromDouble(collisions[i].second->x));
            PyTuple_SetItem(p2_tuple, 1, PyFloat_FromDouble(collisions[i].second->y));
            if (collisions[i].second->data) {
                Py_INCREF(collisions[i].second->data);
                PyTuple_SetItem(p2_tuple, 2, collisions[i].second->data);
            }
            
            PyDict_SetItemString(collision_dict, "point1", p1_tuple);
            PyDict_SetItemString(collision_dict, "point2", p2_tuple);
            
            Py_DECREF(p1_tuple);
            Py_DECREF(p2_tuple);
            
            PyList_SetItem(result_list, i, collision_dict);
        }
        
        return result_list;
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    }
}

static PyObject* quadtree_size(QuadTreeObject* self, PyObject* Py_UNUSED(ignored)) {
    if (!self->qtree) {
        PyErr_SetString(PyExc_RuntimeError, "QuadTree not initialized");
        return nullptr;
    }
    
    return PyLong_FromSize_t(self->qtree->size());
}

static PyObject* quadtree_empty(QuadTreeObject* self, PyObject* Py_UNUSED(ignored)) {
    if (!self->qtree) {
        PyErr_SetString(PyExc_RuntimeError, "QuadTree not initialized");
        return nullptr;
    }
    
    return PyBool_FromLong(self->qtree->empty());
}

static PyObject* quadtree_depth(QuadTreeObject* self, PyObject* Py_UNUSED(ignored)) {
    if (!self->qtree) {
        PyErr_SetString(PyExc_RuntimeError, "QuadTree not initialized");
        return nullptr;
    }
    
    return PyLong_FromSize_t(self->qtree->depth());
}

static PyObject* quadtree_boundary(QuadTreeObject* self, PyObject* Py_UNUSED(ignored)) {
    if (!self->qtree) {
        PyErr_SetString(PyExc_RuntimeError, "QuadTree not initialized");
        return nullptr;
    }
    
    const auto& boundary = self->qtree->boundary();
    PyObject* tuple = PyTuple_New(4);
    if (!tuple) {
        return nullptr;
    }
    
    PyTuple_SetItem(tuple, 0, PyFloat_FromDouble(boundary.x));
    PyTuple_SetItem(tuple, 1, PyFloat_FromDouble(boundary.y));
    PyTuple_SetItem(tuple, 2, PyFloat_FromDouble(boundary.width));
    PyTuple_SetItem(tuple, 3, PyFloat_FromDouble(boundary.height));
    
    return tuple;
}

static PyMethodDef quadtree_methods[] = {
    {"insert", reinterpret_cast<PyCFunction>(quadtree_insert), METH_VARARGS, 
     "Insert a point (x, y) or (x, y, data) into the quadtree"},
    {"query", reinterpret_cast<PyCFunction>(quadtree_query), METH_VARARGS, 
     "Query points in rectangular region (x, y, width, height)"},
    {"detect_collisions", reinterpret_cast<PyCFunction>(quadtree_detect_collisions), METH_VARARGS, 
     "Detect collisions within given radius"},
    {"get_all_points", reinterpret_cast<PyCFunction>(quadtree_get_all_points), METH_NOARGS, 
     "Get all points in the quadtree"},
    {"contains", reinterpret_cast<PyCFunction>(quadtree_contains), METH_VARARGS, 
     "Check if a point (x, y) exists"},
    {"size", reinterpret_cast<PyCFunction>(quadtree_size), METH_NOARGS, 
     "Get total number of points"},
    {"empty", reinterpret_cast<PyCFunction>(quadtree_empty), METH_NOARGS, 
     "Check if quadtree is empty"},
    {"depth", reinterpret_cast<PyCFunction>(quadtree_depth), METH_NOARGS, 
     "Get maximum depth of the quadtree"},
    {"boundary", reinterpret_cast<PyCFunction>(quadtree_boundary), METH_NOARGS, 
     "Get boundary as (x, y, width, height) tuple"},
    {"subdivisions", reinterpret_cast<PyCFunction>(quadtree_subdivisions), METH_NOARGS, 
     "Get number of subdivisions"},
    {nullptr}
};

static PyTypeObject QuadTreeType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "quadtree.QuadTree",           /* tp_name */
    sizeof(QuadTreeObject),        /* tp_basicsize */
    0,                            /* tp_itemsize */
    (destructor)quadtree_dealloc, /* tp_dealloc */
    0,                            /* tp_vectorcall_offset */
    0,                            /* tp_getattr */
    0,                            /* tp_setattr */
    0,                            /* tp_as_async */
    0,                            /* tp_repr */
    0,                            /* tp_as_number */
    0,                            /* tp_as_sequence */
    0,                            /* tp_as_mapping */
    0,                            /* tp_hash  */
    0,                            /* tp_call */
    0,                            /* tp_str */
    0,                            /* tp_getattro */
    0,                            /* tp_setattro */
    0,                            /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    "QuadTree spatial data structure for efficient 2D point queries", /* tp_doc */
    0,                            /* tp_traverse */
    0,                            /* tp_clear */
    0,                            /* tp_richcompare */
    0,                            /* tp_weaklistoffset */
    0,                            /* tp_iter */
    0,                            /* tp_iternext */
    quadtree_methods,             /* tp_methods */
    0,                            /* tp_members */
    0,                            /* tp_getset */
    0,                            /* tp_base */
    0,                            /* tp_dict */
    0,                            /* tp_descr_get */
    0,                            /* tp_descr_set */
    0,                            /* tp_dictoffset */
    (initproc)quadtree_init,      /* tp_init */
    0,                            /* tp_alloc */
    quadtree_new,                 /* tp_new */
};

static PyModuleDef quadtree_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "quadtree",
    .m_doc = "Efficient spatial quadtree data structure for 2D point operations",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_quadtree(void) {
    if (PyType_Ready(&QuadTreeType) < 0) {
        return nullptr;
    }
    
    PyObject* m = PyModule_Create(&quadtree_module);
    if (!m) {
        return nullptr;
    }
    
    Py_INCREF(&QuadTreeType);
    if (PyModule_AddObject(m, "QuadTree", reinterpret_cast<PyObject*>(&QuadTreeType)) < 0) {
        Py_DECREF(&QuadTreeType);
        Py_DECREF(m);
        return nullptr;
    }
    
    return m;
}