#ifndef SOKUDO_ERRORS_H
#define SOKUDO_ERRORS_H

#include <exception>
#include <string>

namespace sokudo::errors {
    class InvalidOperationException : public std::exception {
    private:
        std::string msg;
    public:
        InvalidOperationException(const std::string &msg) {
            this->msg = "sokudo: " + msg;
        }

        [[nodiscard]] const char *what() const noexcept override {
            return msg.c_str();
        }
    };
}

#endif
